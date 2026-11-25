"""Exact original-token offsets for fast tokenizers and byte-level BPE outputs.

No token sequence is replaced by a canonical re-encoding. UTF-8 split tokens may
overlap in character offsets, as with Hugging Face fast-tokenizer offsets.
"""
from bisect import bisect_left, bisect_right
import json


def _byte_decoder():
    values = list(range(ord('!'), ord('~')+1)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256+sum(c >= 256 for c in chars))
    return {chr(char): byte for byte, char in zip(values, chars)}


def _decoded_char_spans(data):
    """Match Python UTF-8 errors=replace, retaining consumed byte spans."""
    chars, spans, cursor = [], [], 0
    while cursor < len(data):
        try:
            valid = data[cursor:].decode('utf-8', errors='strict')
            invalid = None
        except UnicodeDecodeError as error:
            valid = data[cursor:cursor+error.start].decode('utf-8', errors='strict')
            invalid = (cursor+error.start, cursor+error.end)
        for char in valid:
            stop = cursor+len(char.encode('utf-8'))
            chars.append(char)
            spans.append((cursor, stop))
            cursor = stop
        if invalid is not None:
            chars.append('\ufffd')
            spans.append(invalid)
            cursor = invalid[1]
    return ''.join(chars), spans


def original_token_offsets(tokenizer, token_ids):
    token_ids = list(token_ids)
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded['input_ids'] if isinstance(encoded, dict) else encoded.input_ids)
    offsets = encoded['offset_mapping'] if isinstance(encoded, dict) else encoded.offset_mapping
    if ids == token_ids:
        return text, list(map(tuple, offsets)), {'method':'identical_roundtrip', 'n_tokens':len(token_ids)}
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    decoder = getattr(backend, 'decoder', None)
    try:
        decoder_config = json.loads(decoder.__getstate__())
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Noncanonical tokenization requires a verifiable ByteLevel decoder') from error
    if decoder_config.get('type') != 'ByteLevel':
        raise ValueError('Noncanonical offsets currently support ByteLevel decoders only')
    inverse = _byte_decoder()
    special = set(getattr(tokenizer, 'all_special_ids', []))
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    byte_parts = []
    for token_id, piece in zip(token_ids, pieces):
        if token_id in special:
            byte_parts.append(piece.encode('utf-8'))
        else:
            try:
                byte_parts.append(bytes(inverse[c] for c in piece))
            except KeyError as error:
                raise ValueError(f'Unsupported ByteLevel/added token {token_id}') from error
    rebuilt, char_spans = _decoded_char_spans(b''.join(byte_parts))
    if rebuilt != text:
        raise ValueError('Original ByteLevel bytes do not reconstruct decoded text exactly')
    starts, ends = [a for a, _ in char_spans], [b for _, b in char_spans]
    offsets, cursor = [], 0
    for part in byte_parts:
        stop = cursor+len(part)
        offsets.append((bisect_right(ends, cursor), bisect_left(starts, stop)))
        cursor = stop
    if len(offsets) != len(token_ids):
        raise ValueError('Offset count differs from original tokens')
    return text, offsets, {'method':'original_bytelevel_bytes', 'n_tokens':len(token_ids),
                           'n_reencoded_tokens':len(ids)}
