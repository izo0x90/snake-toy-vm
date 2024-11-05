import logging
from typing import Sequence

import vm_types

logger = logging.getLogger(__name__)


class Assembler:
    def __init__(
        self,
        program_text,
        instruction_codes: type[vm_types.GenericInstructionSet],
        word_size,
        instructions_meta,
        macros_meta,
    ):
        self.text = program_text
        self.codes = instruction_codes
        self.instructions_meta = instructions_meta
        self.macros_meta = macros_meta
        self.word_size = word_size
        self.word_size_bytes = word_size // vm_types.BITS_IN_BYTE
        self._reset_state()

    def _reset_state(self):
        self.symbol_table = {"map": {}, "refs": {}}

    def load_program(self, program_text):
        self._reset_state()
        self.text = program_text
        self.byte_code = bytearray()

    def tokenize(self) -> Sequence[vm_types.AssemblerToken]:
        partial_split_lines = self.text.split("\n")
        label_split_code_lines = []
        for line in partial_split_lines:
            label_split_code_lines.extend(line.split(":"))

        code_lines = []
        for line in label_split_code_lines:
            potential_code = line.split("#", 1)[0].strip()
            if potential_code:
                code_lines.append(potential_code)

        tokens = []
        for line in code_lines:
            tokens.extend(line.split(" "))

        assembler_tokens = []
        for token in (tokens := filter(lambda x: x, tokens)):
            if code := getattr(self.codes, token, None):
                meta = self.instructions_meta[code]
                param_values = []
                for param_type in meta.params:
                    param_values.append(param_type(value=next(tokens)))
                assembler_tokens.append(meta.ins(code=code, params=param_values))
            elif token in self.macros_meta:
                token_class, meta = self.macros_meta[token]
                param_values = []
                for param_type in meta.params:
                    next_token = next(tokens)
                    param_values.append(param_type(value=next_token))

                assembler_tokens.append(token_class(params=param_values))

        logger.debug(assembler_tokens)
        return assembler_tokens

    def assemble(self):
        self.byte_code = bytearray()
        for token in self.tokenize():
            if code_bytes := token.encode(assembler_instance=self):
                self.byte_code.extend(code_bytes)

        logger.debug(self.byte_code)
        logger.debug(self.symbol_table)
        return self.byte_code

    def link(self):
        for ref_label, ref_locations in self.symbol_table["refs"].items():
            for location in ref_locations:
                symbol_data = self.symbol_table["map"][ref_label]
                for idx, singe_byte in enumerate(
                    symbol_data.to_bytes(length=self.word_size_bytes)
                ):
                    self.byte_code[location + idx] = singe_byte

        return self.byte_code

    def compile(self):
        self.assemble()
        self.link()
        return self.byte_code
