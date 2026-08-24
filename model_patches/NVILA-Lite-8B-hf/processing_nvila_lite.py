import re
from typing import cast


import numpy as np
import transformers.image_transforms as image_transforms
import transformers.image_utils as image_utils
try:
    import transformers.video_utils as video_utils
except ImportError:
    video_utils = None
from PIL.Image import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen2 import Qwen2Tokenizer, Qwen2TokenizerFast
from transformers.models.siglip import SiglipImageProcessor, SiglipImageProcessorFast
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import BatchEncoding, TextInput
try:
    from transformers.video_utils import VideoInput, VideoMetadata
except ImportError:
    from typing import Any as VideoInput
    from typing import Any as VideoMetadata



class NVILALiteProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}  # type: ignore



class NVILALiteProcessor(ProcessorMixin):
    attributes = [
        "image_processor",
        "tokenizer",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    _auto_class = "AutoProcessor"


    def __init__(
        self,
        image_processor: SiglipImageProcessor | SiglipImageProcessorFast,
        tokenizer: Qwen2Tokenizer | Qwen2TokenizerFast,
        chat_template: str | None = None,
        **kwargs,
    ):
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            **kwargs,
        )


        self.image_processor: SiglipImageProcessor | SiglipImageProcessorFast
        self.tokenizer: Qwen2Tokenizer | Qwen2TokenizerFast


    def __call__(
        self,
        *,
        text: TextInput | list[TextInput],
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[NVILALiteProcessorKwargs],
    ) -> BatchFeature:
        normalized_text, normalized_images, normalized_videos = self._normalize_inputs(
            text=text,
            images=images,
            videos=videos,
        )


        images_inputs, image_token_padding_strategy = (
            self._preprocess_images(
                normalized_images,
                **kwargs,
            )
            if len(normalized_images) > 0
            else (BatchFeature(), [])
        )


        videos_inputs, video_token_padding_strategy = (
            self._preprocess_videos(
                normalized_videos,
                **kwargs,
            )
            if len(normalized_videos) > 0
            else (BatchFeature(), [])
        )


        text_inputs = self._preprocess_text(
            normalized_text,
            image_token_padding_strategy=image_token_padding_strategy,
            video_token_padding_strategy=video_token_padding_strategy,
            **kwargs,
        )


        return BatchFeature(
            {
                **text_inputs,
                **images_inputs,
                **videos_inputs,
            }
        )


    def batch_decode(self, *args, **kwargs) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)


    def _normalize_inputs(
        self,
        *,
        text: TextInput | list[TextInput],
        images: ImageInput | None,
        videos: VideoInput | None,
    ) -> tuple[list[str], list[Image], list[list[Image]]]:
        if isinstance(text, list):
            normalized_text = text
        else:
            normalized_text = [text]


        if images is not None and images != []:
            image_flat_list = cast(list, image_utils.make_flat_list_of_images(images))
            normalized_images = [cast(Image, image_transforms.to_pil_image(image)) for image in image_flat_list]
        else:
            normalized_images = []


        if videos is not None and videos != [] and video_utils is not None:
            video_list = cast(list[list], video_utils.make_batched_videos(videos))
            normalized_videos = [
                [cast(Image, image_transforms.to_pil_image(image)) for image in video] for video in video_list
            ]
        else:
            normalized_videos = []


        return normalized_text, normalized_images, normalized_videos


    def _preprocess_images(
        self,
        images: list[Image],
        **kwargs: Unpack[NVILALiteProcessorKwargs],
    ) -> tuple[BatchFeature, list[list[int]]]:
        merged_kwargs = self._merge_kwargs(
            NVILALiteProcessorKwargs,  # type: ignore
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )


        images = [image.convert("RGB") for image in images]


        if len(images) == 1:
            assert self.image_processor.size["height"] == self.image_processor.size["width"]


            image_tiles = dynamic_preprocess(
                images[0],
                min_num=1,
                max_num=12,
                image_size=self.image_processor.size["height"],
            )


            pixel_values = self.image_processor(
                image_tiles,
                **merged_kwargs["images_kwargs"],
            )["pixel_values"]


            images_inputs = BatchFeature(
                {
                    "pixel_values": pixel_values,
                }
            )


            padding_strategy = [[121] * len(image_tiles)]


        else:
            pixel_values = self.image_processor(
                images,
                **merged_kwargs["images_kwargs"],
            )["pixel_values"]


            images_inputs = BatchFeature(
                {
                    "pixel_values": pixel_values,
                }
            )


            padding_strategy = [[121]] * len(images)


        return images_inputs, padding_strategy


    def _preprocess_text(
        self,
        text: list[str],
        *,
        image_token_padding_strategy: list[list[int]],
        video_token_padding_strategy: list[list[int]],
        **kwargs: Unpack[NVILALiteProcessorKwargs],
    ) -> BatchEncoding:
        assert isinstance(self.tokenizer.image_token, str)
        assert isinstance(self.tokenizer.video_token, str)


        for media_token, padding_strategy in (
            (self.tokenizer.image_token, image_token_padding_strategy),
            (self.tokenizer.video_token, video_token_padding_strategy),
        ):
            assert sum([s.count(media_token) for s in text]) == len(padding_strategy)


            pad_lens = [len(x) for x in padding_strategy]
            text = [re.sub(rf"({re.escape(media_token)})", lambda _: media_token * pad_lens.pop(0), s) for s in text]


            if len(image_token_padding_strategy) == 1 and media_token == self.tokenizer.image_token:
                image_token = self.tokenizer.image_token
                assert isinstance(image_token, str)


                text = [re.sub(rf"({re.escape(image_token)})", r"\1\n", s) for s in text]


            pad_lens = [y for x in padding_strategy for y in x]
            pad_lens = [x + 1 for x in pad_lens]
            text = [re.sub(rf"({re.escape(media_token)})", lambda _: media_token * pad_lens.pop(0), s) for s in text]


        merged_kwargs = self._merge_kwargs(
            NVILALiteProcessorKwargs,  # type: ignore
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )


        text_inputs = self.tokenizer(
            text=text,
            **merged_kwargs["text_kwargs"],
        )


        lf_token_id = self.tokenizer.encode("\n")[0]
        assert isinstance(self.tokenizer.image_token_id, int)
        assert isinstance(self.tokenizer.video_token_id, int)


        input_ids = text_inputs.input_ids


        for media_token_id, padding_strategy in [
            (self.tokenizer.image_token_id, image_token_padding_strategy),
            (self.tokenizer.video_token_id, video_token_padding_strategy),
        ]:
            pad_lens = [y for x in padding_strategy for y in x]


            for i in range(len(input_ids)):
                j = 0
                while j < len(input_ids[i]):
                    if input_ids[i][j] != media_token_id:
                        j += 1
                        continue


                    j += pad_lens.pop(0)
                    input_ids[i][j] = lf_token_id


                    j += 1


        return text_inputs


    def _preprocess_videos(
        self,
        videos: list[list[Image]],
        **kwargs: Unpack[NVILALiteProcessorKwargs],
    ) -> tuple[BatchFeature, list[list[int]]]:
        merged_kwargs = self._merge_kwargs(
            NVILALiteProcessorKwargs,  # type: ignore
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )


        if merged_kwargs["videos_kwargs"].get("do_sample_frames"):
            videos = [
                self._sample_frames(
                    video,
                    **merged_kwargs["videos_kwargs"],
                )
                for video in videos
            ]


        videos = [[image.convert("RGB") for image in video] for video in videos]


        frames = [image for video in videos for image in video]
        pixel_values_videos = self.image_processor(
            frames,
            **merged_kwargs["images_kwargs"],
        )["pixel_values"]


        videos_inputs = BatchFeature(
            {
                "pixel_values_videos": pixel_values_videos,
            }
        )


        padding_strategy = [[121] * len(video) for video in videos]


        return videos_inputs, padding_strategy


    def _sample_frames(
        self,
        video: list[Image],
        **kwargs: Unpack[VideosKwargs],
    ) -> list[Image]:
        fps = kwargs.get("fps")
        num_frames = kwargs.get("num_frames")


        if num_frames is not None and fps is None:
            indices = np.round(np.linspace(0, len(video) - 1, num_frames)).astype(int)


            return [video[i] for i in indices]


        elif num_frames is None and fps is not None:
            video_metadata = kwargs.get("video_metadata")


            if isinstance(video_metadata, VideoMetadata):
                total_num_frames = video_metadata.total_num_frames
                duration = video_metadata.duration


            elif isinstance(video_metadata, dict):
                total_num_frames = video_metadata.get("total_num_frames")
                duration = video_metadata.get("duration")


                assert total_num_frames is not None
                assert duration is not None


            else:
                raise NotImplementedError


            indices = np.round(np.linspace(0, total_num_frames - 1, int(fps * duration))).astype(int)


            return [video[i] for i in indices]


        else:
            raise NotImplementedError



def dynamic_preprocess(
    image: Image, min_num: int, max_num: int, image_size: int, use_thumbnail: bool = True
) -> list[Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height


    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])


    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)


    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]


    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images



def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
