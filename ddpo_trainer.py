from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo.prompts
from ddpo.stat_tracking import PerPromptStatTracker
from ddpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import torch.nn.functional as F
import torchvision
import wandb
import random
from functools import partial
import tqdm
import tempfile
from PIL import Image

from omegaconf import DictConfig, OmegaConf

class DDPOTrainer:
    def __init__(self, config : DictConfig, logger, accelerator=None):
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        # else:
        #     config.run_name += "_" + unique_id

        if config.resume_from:
            config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
            if "checkpoint_" not in os.path.basename(config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {config.resume_from}")
                config.resume_from = os.path.join(
                    config.resume_from,
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
                )

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(config.sample_num_steps * config.train_timestep_fraction)

        os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
        print("\nACCELERATE_USE_DEEPSPEED: ", os.environ.get("ACCELERATE_USE_DEEPSPEED"))
        
        # if accelerator is not provided, create a new one and set up tracker
        if accelerator is None:
            accelerator_config = ProjectConfiguration(
                project_dir=os.path.join(config.logdir, config.run_name),
                automatic_checkpoint_naming=True,
                total_limit=config.num_checkpoint_limit,
            )
            self.accelerator = Accelerator(
                log_with="wandb",
                mixed_precision=config.mixed_precision,
                project_config=accelerator_config,
                # we always accumulate gradients across timesteps; we want config.train_gradient_accumulation_steps to be the
                # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
                # the total number of optimizer steps to accumulate across.
                gradient_accumulation_steps=config.train_gradient_accumulation_steps*config.train_num_update)
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers(
                    # project_name="active-diffusion",
                    project_name=config.project_name,
                    # config=config.to_dict(),
                    config=dict(config),
                    init_kwargs={"wandb": {"name": config.run_name}},
                )

        else:
            self.accelerator = accelerator
        logger.info(f"\n{config}")

        # set seed (device_specific is very important to get different prompts on different devices)
        set_seed(config.seed, device_specific=True)

        # load scheduler, tokenizer and models.
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained_model, revision=config.pretrained_revision
        )
        # freeze parameters of models to save more memory
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(not config.use_lora)
        # disable safety checker
        pipeline.safety_checker = None
        # make the progress bar nicer
        pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        # switch to DDIM scheduler
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to inference_dtype
        pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        if config.use_lora:
            pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        if config.use_lora:
            # Set correct lora layers
            lora_attn_procs = {}
            for name in pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = pipeline.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            pipeline.unet.set_attn_processor(lora_attn_procs)

            # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
            # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
            # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
            class _Wrapper(AttnProcsLayers):
                def forward(self, *args, **kwargs):
                    return pipeline.unet(*args, **kwargs)

            unet = _Wrapper(pipeline.unet.attn_processors)
        else:
            unet = pipeline.unet

        self.pipeline = pipeline
        # set up diffusers-friendly checkpoint saving with Accelerate

        def save_model_hook(models, weights, output_dir):
            # TODO #
            # simply close the assertion
            # assert len(models) == 1
            if config.use_lora and isinstance(models[0], AttnProcsLayers):
                self.pipeline.unet.save_attn_procs(output_dir)
            elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
                models[0].save_pretrained(os.path.join(output_dir, "unet"))
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            if config.use_lora and isinstance(models[0], AttnProcsLayers):
                # self.pipeline.unet.load_attn_procs(input_dir)
                tmp_unet = UNet2DConditionModel.from_pretrained(
                    config.pretrained_model,
                    revision=config.pretrained_revision,
                    subfolder="unet",
                )
                tmp_unet.load_attn_procs(input_dir)
                models[0].load_state_dict(
                    AttnProcsLayers(tmp_unet.attn_processors).state_dict()
                )
                del tmp_unet
            elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                del load_model
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            models.pop()  # ensures that accelerate doesn't try to handle loading of the model

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize the optimizer
        if config.train_use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            unet.parameters(),
            lr=config.train_learning_rate,
            betas=(config.train_adam_beta1, config.train_adam_beta2),
            weight_decay=config.train_adam_weight_decay,
            eps=config.train_adam_epsilon,
        )

        # prepare prompt fn
        self.prompt_fn = getattr(ddpo.prompts, config.prompt_fn)

        # generate negative prompt embeddings
        # NOTE: can define negative prompt here
        neg_prompt_embed = self.pipeline.text_encoder(
            self.pipeline.tokenizer(
                [config.negative_prompt],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample_batch_size, 1, 1)
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train_batch_size, 1, 1)

        # initialize stat tracker
        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast
        # self.autocast = accelerator.autocast

        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer = self.accelerator.prepare(unet, optimizer) 

        # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
        # remote server running llava inference.
        self.executor = futures.ThreadPoolExecutor(max_workers=4)

        # sampling settings and sanity check
        self.samples = {}
        self.samples_per_epoch = (
            config.sample_batch_size
            * self.accelerator.num_processes
            * config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            config.train_batch_size
            * self.accelerator.num_processes
            * config.train_gradient_accumulation_steps
        )

        assert config.sample_batch_size >= config.train_batch_size
        assert config.sample_batch_size % config.train_batch_size == 0
        assert self.samples_per_epoch % self.total_train_batch_size == 0

        self.samples = []

        # store config
        self.config = config

        if self.config.resume_from:
            logger.info(f"Resuming from {self.config.resume_from}")
            self.accelerator.load_state(self.config.resume_from)

        self.global_step = 0

        # set up tqdm
        self.tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


    # NOTE: Remove reward_model and processor args because reward calculation
    # is move to train()
    def sample(self,
               logger,
               epoch,
               save_images=False,
               img_save_dir="sampled_images", 
               high_reward_latents=None):
        self.pipeline.unet.eval()
        self.samples = []
        self.prompts = []
        all_text_embeds = []
        sample_index = 0
        
        for i in self.tqdm(
            range(self.config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling", # TODO
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            # original prompts function that sample a prompt at a time
            prompts, prompt_metadata = zip(
                *[
                    self.prompt_fn()
                    for _ in range(self.config.sample_batch_size)
                ]
            )
            # encode prompts
            prompt_ids = self.pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]
            all_text_embeds.append(prompt_embeds)
            # Add a small perturbation on the high reward latents
            if high_reward_latents is not None:
                condition_latents = []
                ### best and good with fixed noise ###
                if self.config.sample_latent_type == "best_and_good_fixed_noise":
                    print("sampling with best image ratio", self.config.best_latent_ratio)
                    num_best_sample = self.config.sample_num_batches_per_epoch * self.config.sample_batch_size * self.config.best_latent_ratio
                    for i in range(self.config.sample_batch_size):
                        if sample_index < num_best_sample:
                            condition_latents.append(high_reward_latents[-1].unsqueeze(0))
                        elif num_best_sample == 0:
                            pos_index = random.choice(np.arange(high_reward_latents.shape[0]))
                            condition_latents.append(high_reward_latents[pos_index].unsqueeze(0))
                        else:
                            pos_index = random.choice(np.arange(high_reward_latents.shape[0]-1))
                            condition_latents.append(high_reward_latents[pos_index].unsqueeze(0))
                        sample_index += 1 # sample index from 0 to total_batch_size - 1 (sample_batch_size * num_bat)
                    condition_latents = torch.cat(condition_latents)
                    # add noise on all latents
                    noise = torch.Tensor([self.config.latent_noise]).to(self.accelerator.device)
                    alpha = torch.sqrt(1-noise*noise).to(self.accelerator.device)   
                    condition_latents = alpha*condition_latents + noise*torch.randn(condition_latents.shape).to(self.accelerator.device)
    
                condition_latents = condition_latents.half()
            else:
                condition_latents = None
 
            # sample
            with self.autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    self.pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    latents=condition_latents,
                    output_type="pt",
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.pipeline.scheduler.timesteps.repeat(
                self.config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            # store the list form to self.samples
            self.samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "images": images,
                }
            )
            self.prompts.append(prompts)

        # return tensor of images
        samples = torch.cat([sample["images"] for sample in self.samples])
        features = torch.cat([sample["latents"] for sample in self.samples])

        if save_images and self.accelerator.is_main_process:
            self.save_batch_to_images(image_batch=samples, epoch=epoch, save_dir=img_save_dir)

        print(f'Prompts: {self.prompts}')
        return samples, features


    def train_from_reward_labels(self, raw_rewards, logger, epoch):
        """
        Takes raw reward values as input rather than computing with a reward model
        Args:
            raw_rewards () : 
        """
        # TODO logging

        # Compute rewards using most recent reward model
        for i in self.tqdm(
            range(self.config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling", # TODO
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            rewards = raw_rewards[i*self.config["sample_batch_size"] : (i+1)*self.config["sample_batch_size"]]
            rewards = torch.as_tensor(rewards, device=self.accelerator.device)
            # yield to to make sure reward computation starts
            time.sleep(0)
    
            self.samples[i]["rewards"] = rewards

        self.samples = {k: torch.cat([s[k] for s in self.samples]) for k in self.samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(self.samples["images"][-self.config.sample_batch_size:]):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                # pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            self.accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(self.prompts[-1], rewards)
                        )  # only log rewards from process 0
                    ],
                },
                # step=self.global_step,
            )

        # gather rewards across processes
        rewards = self.accelerator.gather(self.samples["rewards"]).cpu().numpy()

        # log rewards and images
        self.accelerator.log(
            {
                "ddpo_epoch": epoch,
                "ddpo_reward_mean": rewards.mean(),
                "ddpo_reward_std": rewards.std(),
            },
            # step=self.global_step,
        )

        # per-prompt mean/std tracking
        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(self.samples["prompt_ids"]).cpu().numpy()
            prompts = self.pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        self.samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del self.samples["rewards"]
        del self.samples["prompt_ids"]

        total_batch_size, num_timesteps = self.samples["timesteps"].shape
        assert (
            total_batch_size
            == self.config.sample_batch_size * self.config.sample_num_batches_per_epoch
        )
        assert num_timesteps == self.config.sample_num_steps

        #################### TRAINING ####################
        for inner_epoch in range(self.config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            self.samples = {k: v[perm] for k, v in self.samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=self.accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                self.samples[key] = self.samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, self.config.train_batch_size, *v.shape[1:])
                for k, v in self.samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            self.pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in self.tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                if self.config.train_cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [self.train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                
                for k in self.tqdm(
                    range(self.config.train_num_update),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                ):
                    j = num_timesteps - 1 - k # remove randomness for reproducibility

                    with self.accelerator.accumulate(self.unet):
                        with self.autocast():
                            if self.config.train_cfg:
                                noise_pred = self.unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + self.config.sample_guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = self.unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                self.pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=self.config.sample_eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -self.config.train_adv_clip_max,
                            self.config.train_adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        info["ddpo_ratio"].append(ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - self.config.train_clip_range,
                            1.0 + self.config.train_clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["ddpo_approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["ddpo_clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > self.config.train_clip_range
                                ).float()
                            )
                        )
                        info["ddpo_loss"].append(loss)

                        # backward pass
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.unet.parameters(), self.config.train_max_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        # assert (j == num_train_timesteps - 1) and (
                            # i + 1
                        # ) % config.train_gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = self.accelerator.reduce(info, reduction="mean")
                        info.update({"ddpo_epoch": epoch, "ddpo_inner_epoch": inner_epoch})
                        info.update({"ddpo_step": self.global_step})
                        # self.accelerator.log(info, step=self.global_step)
                        self.accelerator.log(info)
                        self.global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert self.accelerator.sync_gradients
        # TODO #
        if self.config.save_freq > 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()
        return epoch


    def save_batch_to_images(self, image_batch, epoch, save_dir):
        save_folder = os.path.join(save_dir, f"epoch{epoch}")
        print(f"saving images to {save_folder}")
        os.mkdir(save_folder)
        for i, im in enumerate(image_batch):
            pil_im = torchvision.transforms.functional.to_pil_image(im)
            pil_im.save(os.path.join(save_folder, f"{i}.jpg"))
