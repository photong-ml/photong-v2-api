"""
This module contains the pipeline for Photong.
"""

from pathlib import Path
from typing import Any, Optional, Literal

import note_seq
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from magenta.models import music_vae

from photong.types import TonalityType
from photong.utils.mel_util import touch_up


class PhotongConfig:
    """
    Configuration for Photong pipeline.

    Attributes:
        decoder_config (str): The decoder configuration to use.
            Defaults to "hierdec-mel_16bar".
        checkpoint_dir (str): The directory to store checkpoints.
            Defaults to "checkpoints".
        arousal_checkpoint (str): The checkpoint for the arousal model.
            Defaults to "arousal_model_latest.h5".
        valence_checkpoint (str): The checkpoint for the valence model.
            Defaults to "valence_model_latest.h5".
        embedding_checkpoint (str): The checkpoint for the embedding model.
            Defaults to "embedding_model_latest.h5".
        decoder_checkpoint (str): The checkpoint for the decoder model.
            Defaults to "hierdec-mel_16bar.tar".
        inception_model (Optional[tf.keras.Model]): The inception model.
        arousal_model (Optional[tf.keras.Model]): The arousal model.
        embedding_model (Optional[tf.keras.Model]): The embedding model.
        valence_model (Optional[tf.keras.Model]): The valence model.
    """

    # MusicVAE configuration
    decoder_config: str = "hierdec-mel_16bar"

    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    arousal_checkpoint: str = "arousal_model_latest.h5"
    valence_checkpoint: str = "valence_model_latest.h5"
    embedding_checkpoint: str = "embedding_model_latest.h5"
    decoder_checkpoint: str = f"{decoder_config}.tar"

    # Models
    inception_model: Optional[tf.keras.Model] = None
    arousal_model: Optional[tf.keras.Model] = None
    embedding_model: Optional[tf.keras.Model] = None
    valence_model: Optional[tf.keras.Model] = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize PhotongConfig.

        Args:
            **kwargs: Keyword arguments to override the default configuration.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class PhotongPipeline:
    """Main Photong pipeline."""

    def __init__(self, config: Optional[PhotongConfig] = None) -> None:
        """
        Initialize pipeline.

        Args:
            config (Optional[PhotongConfig]): The configuration to use.

        Raises:
            TypeError: If the configuration is not an instance of PhotongConfig.
            TypeError: If any of the models specified in the configuration is invalid.
        """

        if config is None:
            config = PhotongConfig()
        elif not isinstance(config, PhotongConfig):
            raise TypeError("config must be of type PhotongConfig")

        # Load InceptionV3 model
        if config.inception_model is None:
            config.inception_model = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet"
            )
        elif not isinstance(config.inception_model, tf.keras.Model):
            raise TypeError("inception_model must be of type tf.keras.Model")

        self.img_model = tf.keras.Model(
            config.inception_model.input,
            config.inception_model.layers[-1].output,
        )

        # Load arousal model
        if config.arousal_model is None:
            config.arousal_model = tf.keras.models.load_model(
                Path(config.checkpoint_dir, config.arousal_checkpoint)
            )
        elif not isinstance(config.arousal_model, tf.keras.Model):
            raise TypeError("arousal_model must be of type tf.keras.Model")

        self.arousal_model = config.arousal_model

        # Load valence model
        if config.valence_model is None:
            config.valence_model = tf.keras.models.load_model(
                Path(config.checkpoint_dir, config.valence_checkpoint)
            )
        elif not isinstance(config.valence_model, tf.keras.Model):
            raise TypeError("valence_model must be of type tf.keras.Model")

        self.valence_model = config.valence_model

        # Load embedding model
        if config.embedding_model is None:
            config.embedding_model = tf.keras.models.load_model(
                Path(config.checkpoint_dir, config.embedding_checkpoint)
            )
        elif not isinstance(config.embedding_model, tf.keras.Model):
            raise TypeError("embedding_model must be of type tf.keras.Model")

        self.embedding_model = config.embedding_model

        self.__decoder_config = {
            "config": config.decoder_config,
            "checkpoint_dir": config.checkpoint_dir,
            "checkpoint_file": config.decoder_checkpoint,
        }
        self.decoder_model: music_vae.TrainedModel = None
        self.decoder_config: music_vae.Config = None

    def encode(
        self, img_path: Optional[str] = None, img_data: Optional[str] = None
    ) -> npt.NDArray[np.float32]:
        """
        Load image from path, preprocess, and return embedding.

        Args:
            img_path (Optional[str]): The path to an image.
            img_data (Optional[str]): Raw image data in web-safe Base64 encoding.

        Returns:
            np.ndarray[np.float32]: The embedding.

        Raises:
            ValueError: If both img_path and img_data are not provided.
            ValueError: If both img_path and img_data are provided.
        """

        if img_path is None and img_data is None:
            raise ValueError("img_path or img_data must be provided")

        if img_path is not None and img_data is not None:
            raise ValueError("img_path and img_data cannot both be provided")

        if img_path is not None:
            img_data = tf.io.read_file(img_path)
        else:
            img_data = tf.io.decode_base64(img_data)

        img = tf.image.decode_image(img_data, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        img_features: npt.NDArray[np.float32] = self.img_model(img).numpy()

        return img_features

    def get_valence(
        self, img_emb: npt.NDArray[np.float32], *, verbose: Literal["auto", 0, 1, 2] = 0
    ) -> float:
        """
        Get valence from a image embedding.

        Args:
            img_emb (np.ndarray[np.float32]): The image embedding.
            verbose (Literal["auto", 0, 1, 2]): The verbosity level. See tf.keras.Model.predict().
                Defaults to 0.

        Returns:
            float: The valence value.
        """

        return float(
            self.valence_model.predict(img_emb, verbose=verbose).reshape(-1)[0]
        )

    def get_tonality(
        self, img_emb: npt.NDArray[np.float32], *, verbose: Literal["auto", 0, 1, 2] = 0
    ) -> TonalityType:
        """
        Get tonality from a image embedding based on valence.

        Args:
            img_emb (np.ndarray[np.float32]): The image embedding.
            verbose (Literal["auto", 0, 1, 2]): The verbosity level. See tf.keras.Model.predict().
                Defaults to 0.

        Returns:
            TonalityType: The tonality.
        """

        valence = self.get_valence(img_emb, verbose=verbose)
        return TonalityType.MAJOR if valence >= 0.5 else TonalityType.MINOR

    def get_arousal(
        self, img_emb: npt.NDArray[np.float32], *, verbose: Literal["auto", 0, 1, 2] = 0
    ) -> float:
        """
        Get arousal from a image embedding.

        Args:
            img_emb (np.ndarray[np.float32]): The image embedding.
            verbose (Literal["auto", 0, 1, 2]): The verbosity level. See tf.keras.Model.predict().
                Defaults to 0.

        Returns:
            float: The arousal value.
        """

        return float(
            self.arousal_model.predict(img_emb, verbose=verbose).reshape(-1)[0]
        )

    def get_tempo(
        self, img_emb: npt.NDArray[np.float32], *, verbose: Literal["auto", 0, 1, 2] = 0
    ) -> float:
        """
        Get tempo from a image embedding based on arousal.

        Args:
            img_emb (np.ndarray[np.float32]): The image embedding.
            verbose (Literal["auto", 0, 1, 2]): The verbosity level. See tf.keras.Model.predict().
                Defaults to 0.

        Returns:
            float: The tempo value.
        """

        arousal = self.get_arousal(img_emb, verbose=verbose)
        arousal = float(160 * 1 / (1 + np.exp(-5 * (arousal - 0.5))) + 40)
        return arousal

    def get_embedding(
        self, img_emb: npt.NDArray[np.float32], *, verbose: Literal["auto", 0, 1, 2] = 0
    ) -> npt.NDArray[np.float16]:
        """
        Generate a MusicVAE-compatible embedding from a image embedding.

        Args:
            img_emb (np.ndarray[np.float32]): The image embedding.
            verbose (Literal["auto", 0, 1, 2]): The verbosity level. See tf.keras.Model.predict().
                Defaults to 0.

        Returns:
            np.ndarray[np.float16]: A MusicVAE-compatible embedding.
        """

        emb: npt.NDArray[np.float16] = self.embedding_model.predict(
            img_emb, verbose=verbose
        )
        return emb

    def decode(self, aud_emb: npt.NDArray[np.float16]) -> note_seq.NoteSequence:
        """
        Decode a MusicVAE-compatible embedding into a NoteSequence.

        Args:
            aud_emb (np.ndarray[np.float16]): A MusicVAE-compatible embedding.

        Returns:
            note_seq.NoteSequence: The decoded NoteSequence.

        Raises:
            ValueError: If the decoder model is not loaded.
        """

        if self.decoder_model is None:
            # Load decoder model
            self.decoder_config = music_vae.configs.CONFIG_MAP[
                self.__decoder_config["config"]
            ]
            self.decoder_model = music_vae.TrainedModel(
                self.decoder_config,
                batch_size=1,
                checkpoint_dir_or_path=Path(
                    self.__decoder_config["checkpoint_dir"],
                    self.__decoder_config["checkpoint_file"],
                ),
            )

        return self.decoder_model.decode(
            length=self.decoder_config.hparams.max_seq_len,
            z=aud_emb,
            temperature=0.5,
        )[0]

    def predict(
        self,
        img_path: Optional[str] = None,
        img_data: Optional[str] = None,
        *,
        verbose: Literal["auto", 0, 1, 2] = 0,
    ) -> note_seq.NoteSequence:
        """
        Generate a note sequence from an image.

        Args:
            img_path (Optional[str]): The path to an image.
            img_data (Optional[str]): Raw image data in web-safe Base64 encoding.
            verbose (Literal["auto", 0, 1, 2]): The verbosity level. See tf.keras.Model.predict().
                Defaults to 0.

        Returns:
            note_seq.NoteSequence: The generated NoteSequence.
        """

        img_emb = self.encode(img_path, img_data)

        tonality = self.get_tonality(img_emb, verbose=verbose)
        tempo = self.get_tempo(img_emb, verbose=verbose)
        embedding = self.get_embedding(img_emb, verbose=verbose)

        res_ns = self.decode(embedding)
        res_ns = touch_up(res_ns, tonality, tempo)

        return res_ns
