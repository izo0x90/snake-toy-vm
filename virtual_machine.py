from dataclasses import dataclass
import logging

import vm_types

logger = logging.getLogger(__name__)


@dataclass
class VirtualMachine:
    memory: bytearray
    cpu: vm_types.GenericCentralProcessingUnit
    assembler: vm_types.GenericAssembler
    stack_address: int | None = None
    clock_speed_hz: int = 1
    _ellapsed_since_last: float = 0

    def __post_init(self):
        self.restart()

    def get_registers(self):
        return self.cpu.dump_registers()

    def get_program_text(self):
        return self.assembler.text

    def get_current_instruction_address(self):
        return self.cpu.current_inst_address

    def get_video_memory(self):
        # Place holder until video memory gets implemented
        bytes_data = bytearray(320 * 200 * 3)
        for y in range(0, 200):
            for x in range(0, 320, 1):
                row = (320 * 3) * y
                col = x * 3
                bytes_data[row + col] = (x ^ y) % 255
                bytes_data[row + col + 1] = 0
                bytes_data[row + col + 2] = 0

        return bytes_data

    def _replace_memory(self, memory: bytearray):
        self.memory = memory
        self.cpu.RAM = self.memory

    def load_at(self, address: int, data: bytearray, force=False):
        # TODO: I think there is 1 off bug here looking how mem changes
        data_len = len(data)
        memory_len = len(self.memory)

        if not force and address % self.cpu.word_in_bytes != 0:
            raise Exception("TODO")

        if address < 0 or (address + data_len - 1) >= memory_len:
            raise Exception("TODO")

        self._replace_memory(
            self.memory[:address] + data + self.memory[address + data_len - 1 :]
        )

        logger.debug(
            f"Loaded data at {hex(address)=}\n" f"data=({self.memory[:102].hex()}...)"
        )

    def load_program_at(self, address: int, program_text: str):
        self.assembler.load_program(program_text)
        logger.debug(f"Loaded {program_text=}")
        byte_code = self.assembler.compile()
        self.load_at(address, byte_code)

    def load_and_reset(self, program_text: str, address: int = 0):
        self.reset()
        self.load_program_at(address, program_text)
        self.restart()

    def restart(self):
        self.cpu.reset()
        self.cpu.current_stack_address = self.stack_address or len(self.memory) - 1
        self.current_run = self.cpu.run()

    def reset(self):
        self.assembler.load_program("")
        self._replace_memory(bytearray(len(self.memory)))
        self.restart()

    def step(self, milli_sec_since_last: int):
        self._ellapsed_since_last += milli_sec_since_last
        if self._ellapsed_since_last < 1000 / self.clock_speed_hz:
            return

        self._ellapsed_since_last = 0
        try:
            next(self.current_run)
        except StopIteration:
            logger.debug("Current run halted")

    def run(self):
        for _ in self.cpu.run():
            logger.debug("Virtual machine ran step...\n\n")
