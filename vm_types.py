import enum

from typing import (
    Any,
    Callable,
    Generator,
    Mapping,
    Protocol,
    Sequence,
)

BITS_IN_BYTE = 8

GenericInstructionSet = enum.Enum
SizeInBytes = int
RegistersDump = Mapping[str, int]
DecoratorCallable = Callable[[Callable], None]


class GenericAssembler(Protocol):
    word_size_bytes: SizeInBytes
    symbol_table: dict[str, Any]
    byte_code: bytearray
    text: str

    def __init__(
        self,
        program_text: str,
        instruction_codes: type[GenericInstructionSet],
        word_size: SizeInBytes,
    ): ...

    def load_program(self, program_text: str):
        pass

    def tokenize(self) -> Sequence["AssemblerToken"]: ...

    def assemble(self) -> bytearray: ...

    def link(self) -> bytearray: ...

    def compile(self) -> bytearray: ...


class AssemblerToken(Protocol):
    def encode(self, assembler_instance: GenericAssembler) -> bytes: ...


class AssemblerParamToken(Protocol):
    def __init__(self, value: str): ...

    def encode(self, assembler_instance: GenericAssembler, offset: int) -> bytes: ...


class AssemblerMetaParamToken(Protocol):
    def __init__(self, value: str): ...

    def encode(
        self, assembler_instance: GenericAssembler, offset: int
    ) -> AssemblerParamToken: ...


class GenericCentralProcessingUnit(Protocol):
    RAM: bytearray
    # FLAGS: CPUFlags
    # REGISTERS: CPURegisters = field(default_factory=CPURegisters)
    # WORD_SIZE_BITS: int = WORD_SIZE
    # INSTRUCTION_MAP: ClassVar[
    #     MutableMapping[InstructionCodes, Callable[[Self], None]]
    # ] = {}

    @classmethod
    def register_instruction(
        cls, inst_code: GenericInstructionSet
    ) -> DecoratorCallable: ...

    @property
    def word_in_bytes(self) -> int: ...

    @property
    def nible_in_bytes(self) -> int: ...

    def fetch(self): ...

    def reset(self): ...

    def exec(self):
        """Execute one instruction/ step"""

    def run(self) -> Generator:
        """Execute instructions until HALT"""
        ...

    @property
    def current_inst_address(self) -> int:
        """Return the address in the program counter/ instruction pointer"""
        ...

    @property
    def current_stack_address(self) -> int:
        """Return the address in the stack pointer"""
        ...

    @current_stack_address.setter
    def current_stack_address(self, address: int) -> int:
        """Return the address in the stack pointer"""
        ...

    def dump_registers(self) -> RegistersDump:
        """Dump all cpu registers as Mapping object"""
        ...


class GenericVirtualMachine(Protocol):
    memory: bytearray
    cpu: GenericCentralProcessingUnit
    assembler: GenericAssembler
    stack_address: int | None
    clock_speed_hz: int = 1

    def get_registers(self): ...

    def get_program_text(self): ...

    def get_current_instruction_address(self): ...

    def get_video_memory(self): ...

    def load_at(self, address: int, data: bytearray, force=False): ...

    def load_program_at(self, address: int, program_text: str): ...

    def load_and_reset(self, program_text: str, address: int = 0): ...

    def restart(self): ...

    def reset(self): ...

    def step(self, milli_sec_since_last: int): ...

    def run(self): ...
