import enum

from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

BITS_IN_BYTE = 8

GenericInstructionSet = enum.Enum
SizeInBytes = int
RegistersDump = Mapping[str, int]
DecoratorCallable = Callable[[Callable], Callable]


class GenericAssembler(Protocol):
    word_size_bytes: SizeInBytes
    symbol_table: dict[str, Any]
    byte_code: bytearray
    text: str
    instructions_meta: Mapping[str, Callable]
    macros_meta: Mapping[str, Callable]

    def __init__(
        self,
        program_text: str,
        instruction_codes: type[GenericInstructionSet],
        word_size: SizeInBytes,
        instructions_meta: Mapping[str, Callable],
        macros_meta: Mapping[str, Callable],
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
    value: Any

    def __init__(self, value: str): ...

    def encode(
        self, assembler_instance: GenericAssembler, offset: int
    ) -> Union[bytes, bytearray, "AssemblerParamToken"]: ...


class AssemblerMetaParamToken(Protocol):
    def __init__(self, value: str): ...

    def encode(
        self, assembler_instance: GenericAssembler, offset: int
    ) -> AssemblerParamToken: ...


class PortInfo(NamedTuple):
    port_id: int
    read_port: bool


class PortLabeldCallable:
    IS_PORT_HANDLER: ClassVar[str] = "IS_PORT_HANDLER"

    def __init__(self, func: Callable, info: PortInfo):
        self.__func__ = func
        self.info = info

    def __call__(self, *args, **kwargs) -> Any:
        return self.__func__(*args, **kwargs)


class GenericDevice: ...


class VideoResolution(NamedTuple):
    width: int
    heigth: int


class GenericVideoDevice(GenericDevice):
    VRAM: memoryview
    resolution: VideoResolution


class GenericDeviceManager(Protocol):
    video_device: Optional[GenericVideoDevice]
    _WRITE_TO_PORTS: MutableMapping[int, Callable]
    _READ_FROM_PORTS: MutableMapping[int, Callable]

    def read_port(self, port_id: int) -> int: ...

    def write_port(self, port_id: int, value: int): ...


T = TypeVar("T")


class GenericCPURegisterNames(enum.Enum):
    @classmethod
    def add_registers(
        cls: type[enum.Enum],
        cls_to_decorate: type[T],
        reg_type: type = int,
        default: Any = 0,
    ) -> type[T]:
        for reg_name in cls:
            setattr(cls_to_decorate, reg_name.value, default)
            cls_to_decorate.__annotations__[reg_name.value] = reg_type

        return cls_to_decorate

    @classmethod
    def validate_register_names(
        cls: type[enum.Enum], cls_to_decorate: type[T]
    ) -> type[T]:
        for reg_name in cls:
            if not hasattr(cls_to_decorate, reg_name.value):
                raise ValueError("Missing register name={reg_name}")

        return cls_to_decorate


class GenericCentralProcessingUnit(Protocol):
    RAM: memoryview
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


class GenericCentralProcessingUnitDevicePorts(GenericCentralProcessingUnit, Protocol):
    DEVICE_MANAGER: GenericDeviceManager


class GenericVirtualMachine(Protocol):
    memory: memoryview
    cpu: GenericCentralProcessingUnit | GenericCentralProcessingUnitDevicePorts
    assembler: GenericAssembler
    stack_address: int | None
    clock_speed_hz: int = 1
    device_manager: GenericDeviceManager

    def get_registers(self) -> Mapping: ...

    def get_program_text(self) -> str: ...

    def get_current_instruction_address(self) -> int: ...

    @property
    def video_memory(self) -> bytearray | memoryview: ...

    @property
    def video_resolution(self) -> VideoResolution: ...

    def load_at(self, address: int, data: bytearray, force=False): ...

    def load_program_at(self, address: int, program_text: str): ...

    def load_and_reset(self, program_text: str, address: int = 0): ...

    def restart(self): ...

    def reset(self): ...

    def step(self, milli_sec_since_last: int): ...

    def run(self): ...
