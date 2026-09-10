from libevchain.types.artefact_types import (
    expose_artefact_method,
    register_artefact_type,
    ArtefactType,
)
from libevchain.types.attribute import Attribute, AttributeTypes


@register_artefact_type("audio")
class AudioType(ArtefactType):
    '''
        AudioType is the toplevel type for audio files
        it has no special properties
    '''
    audio_creation_methods = AttributeTypes.Literal(
        "recorded",
        "synthesised",
        "ai-generated",
    )

    @staticmethod
    def attributes():
        return {
            # tha manner in which the audio was created,
            # either as a single value or as a list of methods
            'creation_method': Attribute(
                AttributeTypes.Either(
                    audio_creation_method,
                    AttributeTypes.ListOf(audio_creation_method),
                ), "unknown"
            ),
        }



@register_artefact_type("speech")
class SpeechType(AudioType):
    @staticmethod
    def attributes():
        return {
            # the person speaking in the audio file
            'speaker': Attribute(AttributeTypes.Str, None)
        }
