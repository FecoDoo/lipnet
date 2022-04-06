from core.decoding.decoder import Decoder
from core.decoding.spell import Spell
from core.utils.label import labels_to_text
from core.utils.types import PathLike
import env

DECODER_GREEDY = env.DECODER_GREEDY
DECODER_BEAM_WIDTH = env.DECODER_BEAM_WIDTH


def create_decoder(
    dict_path: PathLike, greedy: bool = DECODER_GREEDY, width: int = DECODER_BEAM_WIDTH
):
    spell = Spell(dict_path)
    return Decoder(
        greedy=greedy, beam_width=width, postprocessors=[labels_to_text, spell.sentence]
    )
