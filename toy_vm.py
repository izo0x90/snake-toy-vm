from dataclasses import asdict, field, dataclass
import functools
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    MutableMapping,
    Optional,
    Self,
    Sequence,
    Tuple,
)

from toy_assembler import Assembler
import vm_types
import virtual_machine

logger = logging.getLogger(__name__)

HALT_INS_CODE = 0xFFFF
WORD_SIZE = 16


class InstructionCodes(vm_types.GenericInstructionSet):
    NOP = 0x00
    LOADA = 0x1
    LOADB = 0x2
    LOADC = 0x3
    MLOADA = 0x4
    MLOADB = 0x6
    LOADIX = 0x7
    MLOADIX = 0x8
    JMP = 0x10
    JZ = 0x11
    JO = 0x12
    JS = 0x13
    JC = 0x14
    MOVAIX = 0xA0
    MSTOREA = 0xB0
    MSTOREB = 0xB1
    MSTOREC = 0xB2
    INA = 0xC0
    OUTA = 0xC1
    ADD = 0xF0
    HALT = HALT_INS_CODE


@dataclass
class InlineParamToken:
    value: str

    def encode(
        self, assembler_instance: vm_types.GenericAssembler, offset: int
    ) -> bytes | bytearray:
        word_size_bytes = assembler_instance.word_size_bytes
        signed = False
        match self.value[:2]:
            case "0x":
                base = 16
            case "0b":
                base = 2
            case _:
                base = 10
                if self.value.strip().startswith("-"):
                    signed = True

        return int(self.value, base).to_bytes(signed=signed, length=word_size_bytes)


class MultiInlineParamToken(InlineParamToken):
    value_list: Sequence[str]
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler, offset: int):
        byte_code = bytearray()
        self.value_list = self.value.split(",")
        for str_val in self.value_list:
            self.value = str_val.strip()
            byte_val = super().encode(assembler_instance, offset)
            byte_code.extend(byte_val)

        return byte_code


@dataclass
class LabelParamToken:
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler, offset: int):
        word_size_bytes = assembler_instance.word_size_bytes
        assembler_instance.symbol_table["refs"].setdefault(self.value, []).append(
            len(assembler_instance.byte_code) + offset
        )

        return bytes(word_size_bytes)


@dataclass
class LabelOrInlineParamToken:
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler, offset: int):
        if self.value.startswith("."):
            return LabelParamToken(value=self.value)
        else:
            return InlineParamToken(value=self.value)


@dataclass
class InstructionToken:
    code: InstructionCodes
    params: Sequence[vm_types.AssemblerParamToken]

    def encode(self, assembler_instance: vm_types.GenericAssembler) -> bytes:
        byte_code = bytearray()

        inst_bytes = self.code.value.to_bytes(
            length=assembler_instance.word_size_bytes,
        )

        byte_code.extend(inst_bytes)
        for param_token in self.params:
            bytes_or_token = param_token
            while not (
                isinstance(bytes_or_token, bytes)
                or isinstance(bytes_or_token, bytearray)
            ):
                # TODO: ADD MAX DEPTH ERROR
                bytes_or_token = bytes_or_token.encode(
                    assembler_instance=assembler_instance, offset=len(byte_code)
                )

            byte_code.extend(bytes_or_token)

        return bytes(byte_code)


@dataclass
class InstructionMeta:
    ins: type[vm_types.AssemblerToken] = InstructionToken
    params: Sequence[type[vm_types.AssemblerParamToken]] = field(default_factory=list)


GenericOneInlineParamIns = InstructionMeta(params=[InlineParamToken])
GenericOneLabelOrInlineParamIns = InstructionMeta(params=[LabelOrInlineParamToken])


NOPARAMS_INSTRUCTIONS_META = {
    instruction: InstructionMeta()
    for instruction in [
        InstructionCodes.NOP,
        InstructionCodes.ADD,
        InstructionCodes.HALT,
        InstructionCodes.MOVAIX,
    ]
}


INLINE_PARAM_INSTRUCTIONS_META = {
    instruction: GenericOneInlineParamIns
    for instruction in [
        InstructionCodes.LOADA,
        InstructionCodes.LOADB,
        InstructionCodes.LOADC,
        InstructionCodes.LOADIX,
        InstructionCodes.INA,
        InstructionCodes.OUTA,
    ]
}


ONE_LABEL_INLINE_INSTRUCTIONS_META = {
    instruction: GenericOneLabelOrInlineParamIns
    for instruction in [
        InstructionCodes.MLOADA,
        InstructionCodes.MLOADB,
        InstructionCodes.MLOADIX,
        InstructionCodes.MSTOREA,
        InstructionCodes.MSTOREC,
        InstructionCodes.JMP,
        InstructionCodes.JZ,
        InstructionCodes.JO,
        InstructionCodes.JS,
        InstructionCodes.JC,
    ]
}


INSTRUCTIONS_META = (
    NOPARAMS_INSTRUCTIONS_META
    | INLINE_PARAM_INSTRUCTIONS_META
    | ONE_LABEL_INLINE_INSTRUCTIONS_META
)


@dataclass
class MacroMeta:
    params: Sequence[type[vm_types.AssemblerParamToken]] = field(default_factory=list)


@dataclass
class SetLabelParamToken:
    value: str

    def encode(
        self, assembler_instance: vm_types.GenericAssembler, offset: int
    ) -> bytes:
        return b""


@dataclass
class SetLabelToken:
    params: Sequence[vm_types.AssemblerParamToken]

    def encode(self, assembler_instance: vm_types.GenericAssembler) -> None:
        assembler_instance.symbol_table["map"][self.params[0].value] = len(
            assembler_instance.byte_code
        )


@dataclass
class DefineWordToken:
    params: Sequence[vm_types.AssemblerParamToken]

    def encode(self, assembler_instance: vm_types.GenericAssembler) -> bytes:
        byte_code = bytearray()

        for param_token in self.params:
            bytes_or_token = param_token
            while not (
                isinstance(bytes_or_token, bytes)
                or isinstance(bytes_or_token, bytearray)
            ):
                # TODO: ADD MAX DEPTH ERROR
                bytes_or_token = bytes_or_token.encode(
                    assembler_instance=assembler_instance, offset=len(byte_code)
                )

            byte_code.extend(bytes_or_token)

        return bytes(byte_code)


MACROS_META = {
    "LABEL": (SetLabelToken, MacroMeta(params=[SetLabelParamToken])),
    "DWORD": (DefineWordToken, MacroMeta(params=[MultiInlineParamToken])),
}

TEST_PROG = """
# 16-bit word tests

JMP .START
LABEL .DATA:
DWORD 0xABAB,0xCDCD,0xABAB,0xCDCD,0xABAB

LABEL .START: NOP
# Ports test if RETURN key is pressed
# reg A = 1 if pressed while exec. INA port 10
LOADA 13
OUTA 0x000b
INA 10
LOADB -1
ADD
JZ .END # End program if RETURN pressed during check

# Loop
LOADA 0x000A
LOADB -1
LOADC 0xFFFF

LABEL .VIDEO_LOOP
MOVAIX
MSTOREC 0x1000 # Video memory
ADD
JZ .EXIT_VIDEO_LOOP
JMP .VIDEO_LOOP
LABEL .EXIT_VIDEO_LOOP:

# Write to "data" Loop
LOADA 0x0008
LOADB -1

LABEL .DATA_LOOP
MOVAIX
MSTOREA .DATA
ADD
JZ .EXIT_DATA_LOOP
JMP .DATA_LOOP
LABEL .EXIT_DATA_LOOP:

# Memory loads
MLOADA .END
MLOADA .DATA
LOADIX 0x0001
MLOADB .DATA

# No overflow or carry
LOADA 0xFFFD
LOADB 0x0002
ADD

# Signed overflow and carry
LOADA 0x8000
LOADB 0xffff
ADD

# Ex. Signed overflow 0x7fff (32767) + 0x0001 (1) and signed flag
LOADA 0x7fff
LOADB 0x0001
ADD

LABEL .ONE_ADD:
# Ex. Carry but no Signed overflow 00x8001 (-32767) + 0x7fff (32767) and zero flag
LOADA 0x8001
LOADB 0x7fff
ADD

LABEL .END:
HALT
"""


@dataclass
class AddressModes:
    IMMEDIATE = 0
    DIRECT = 1
    DIRECT_INDEXED = 2
    REGISTER = 255


class CPUFlagNames(vm_types.GenericCPURegisterNames):
    overflow = "overflow"
    carry = "carry"
    signed = "signed"
    zero = "zero"


@dataclass
@CPUFlagNames.validate_register_names
class CPUFlags:
    overflow: bool = False
    carry: bool = False
    signed: bool = False
    zero: bool = False


class CPURegisterNames(vm_types.GenericCPURegisterNames):
    IP = "IP"
    SP = "SP"
    IC = "IC"
    AA = "AA"
    AB = "AB"
    AC = "AC"
    IX = "IX"
    _HIDDEN = "_H"


@dataclass
@CPURegisterNames.validate_register_names
class CPURegisters:
    IP: int = 0
    SP: int = 0
    IC: int = 0
    AA: int = 0
    AB: int = 0
    AC: int = 0
    IX: int = 0
    _H: int = 0


@dataclass
class CentralProcessingUnit:
    HALT_INS_CODE: int
    RAM: memoryview
    FLAGS: CPUFlags
    DEVICE_MANAGER: vm_types.GenericDeviceManager
    REGISTERS: CPURegisters = field(default_factory=CPURegisters)
    WORD_SIZE_BITS: int = WORD_SIZE
    INSTRUCTION_MAP: ClassVar[
        MutableMapping[InstructionCodes, Callable[[Self], None]]
    ] = {}

    @classmethod
    def register_instruction(cls, inst_code) -> vm_types.DecoratorCallable:
        def decorator(f) -> Callable:
            cls.INSTRUCTION_MAP[inst_code] = f
            return f

        return decorator

    @classmethod
    def register_instruction_variants(
        cls, bind_list: Sequence[Tuple[InstructionCodes, dict[str, Any]]]
    ) -> vm_types.DecoratorCallable:
        def decorator(f: Callable) -> Callable:
            for ins_code, kwargs in bind_list:
                bound_func = functools.partial(f, **kwargs)
                cls.register_instruction(ins_code)(bound_func)
            return f

        return decorator

    @property
    def word_in_bytes(self) -> int:
        return self.WORD_SIZE_BITS // vm_types.BITS_IN_BYTE

    @property
    def nible_in_bytes(self) -> int:
        return self.word_in_bytes // 2

    def fetch(self):
        next_ip = self.REGISTERS.IP + self.word_in_bytes
        self.REGISTERS.IC = int.from_bytes(self.RAM[self.REGISTERS.IP : next_ip])
        self.REGISTERS.IP = next_ip

    def reset(self):
        self.REGISTERS = CPURegisters()
        self.FLAGS = CPUFlags()

    def exec(self):
        inst_func = self.INSTRUCTION_MAP[InstructionCodes(self.REGISTERS.IC)]
        inst_func(self)

    def run(self) -> Generator:
        while not self.fetch() and self.REGISTERS.IC != self.HALT_INS_CODE:
            yield

            logger.debug("Running CPU step ...")
            logger.debug(f"{hex(self.REGISTERS.IC)=}")
            self.exec()
            logger.debug(
                f"{InstructionCodes(self.REGISTERS.IC)=}, {self.REGISTERS.AA=}, {self.REGISTERS.AB=}"
            )
            logger.debug(
                f"{InstructionCodes(self.REGISTERS.IC)=}, {hex(self.REGISTERS.AA)=}, {hex(self.REGISTERS.AB)=}"
            )
            logger.debug(f"{self.FLAGS=}")

    def dump_registers(self):
        return asdict(self.REGISTERS)

    @property
    def current_inst_address(self) -> int:
        return self.REGISTERS.IP

    @property
    def current_stack_address(self) -> int:
        return self.REGISTERS.SP

    @current_stack_address.setter
    def current_stack_address(self, address: int) -> int:
        self.REGISTERS.SP = address
        return self.REGISTERS.SP


def load(
    instance: CentralProcessingUnit,
    reg_name: CPURegisterNames,
    ip_unmodified=False,
    address_mode=AddressModes.IMMEDIATE,
):
    bytes_val = instance.RAM[
        instance.REGISTERS.IP : instance.REGISTERS.IP + instance.word_in_bytes
    ]
    val = int.from_bytes(bytes_val)

    match address_mode:
        case AddressModes.IMMEDIATE:
            pass
        case AddressModes.DIRECT_INDEXED:
            address = val + instance.REGISTERS.IX
            val = int.from_bytes(
                instance.RAM[address : address + instance.word_in_bytes]
            )

        case _:
            raise ValueError(f"Unknown {address_mode=} for load OP")

    setattr(instance.REGISTERS, reg_name.value, val)
    if not ip_unmodified:
        instance.REGISTERS.IP += instance.word_in_bytes


def store(
    instance: CentralProcessingUnit,
    source_reg_name: CPURegisterNames,
    *,
    dest_reg_name: Optional[CPURegisterNames] = None,
    address_mode=AddressModes.IMMEDIATE,
):
    val = getattr(instance.REGISTERS, source_reg_name.value)

    match address_mode:
        case AddressModes.DIRECT_INDEXED:
            val = val.to_bytes(length=instance.word_in_bytes)
            address_bytes_val = instance.RAM[
                instance.REGISTERS.IP : instance.REGISTERS.IP + instance.word_in_bytes
            ]
            address = int.from_bytes(address_bytes_val) + instance.REGISTERS.IX

            instance.RAM[address : address + instance.word_in_bytes] = val

            instance.REGISTERS.IP += instance.word_in_bytes
        case AddressModes.REGISTER:
            if not dest_reg_name:
                raise ValueError("Missing destination register name")
            setattr(instance.REGISTERS, dest_reg_name.value, val)

        case _:
            raise ValueError(f"Unknown {address_mode=} for store OP")


def jump(instance: CentralProcessingUnit, flag_name: Optional[CPUFlagNames] = None):
    if not flag_name or getattr(instance.FLAGS, flag_name.value):
        load(instance, CPURegisterNames.IP, ip_unmodified=True)
    else:
        instance.REGISTERS.IP += instance.word_in_bytes


@CentralProcessingUnit.register_instruction(InstructionCodes.NOP)
def noop(instance: CentralProcessingUnit):
    pass


@CentralProcessingUnit.register_instruction_variants(
    [
        (InstructionCodes.LOADA, {"reg_name": CPURegisterNames.AA}),
        (InstructionCodes.LOADB, {"reg_name": CPURegisterNames.AB}),
        (InstructionCodes.LOADC, {"reg_name": CPURegisterNames.AC}),
        (InstructionCodes.LOADIX, {"reg_name": CPURegisterNames.IX}),
    ]
)
def load_immediate(instance: CentralProcessingUnit, reg_name: CPURegisterNames):
    load(instance, reg_name)


@CentralProcessingUnit.register_instruction_variants(
    [
        (InstructionCodes.MLOADA, {"reg_name": CPURegisterNames.AA}),
        (InstructionCodes.MLOADB, {"reg_name": CPURegisterNames.AB}),
        (InstructionCodes.MLOADIX, {"reg_name": CPURegisterNames.IX}),
    ]
)
def mload_direct(instance: CentralProcessingUnit, reg_name: CPURegisterNames):
    load(instance, reg_name, address_mode=AddressModes.DIRECT_INDEXED)


@CentralProcessingUnit.register_instruction_variants(
    [
        (
            InstructionCodes.MOVAIX,
            {
                "source_reg_name": CPURegisterNames.AA,
                "dest_reg_name": CPURegisterNames.IX,
            },
        ),
    ]
)
def rstore(
    instance: CentralProcessingUnit,
    source_reg_name: CPURegisterNames,
    dest_reg_name: CPURegisterNames,
):
    store(
        instance,
        source_reg_name,
        dest_reg_name=dest_reg_name,
        address_mode=AddressModes.REGISTER,
    )


@CentralProcessingUnit.register_instruction_variants(
    [
        (InstructionCodes.MSTOREA, {"source_reg_name": CPURegisterNames.AA}),
        (InstructionCodes.MSTOREC, {"source_reg_name": CPURegisterNames.AC}),
    ]
)
def mstore_direct(instance: CentralProcessingUnit, source_reg_name: CPURegisterNames):
    store(instance, source_reg_name, address_mode=AddressModes.DIRECT_INDEXED)


@CentralProcessingUnit.register_instruction(InstructionCodes.INA)
def in_inst(instance: CentralProcessingUnit):
    load(instance, CPURegisterNames._HIDDEN)
    instance.REGISTERS.AA = instance.DEVICE_MANAGER.read_port(instance.REGISTERS._H)


@CentralProcessingUnit.register_instruction(InstructionCodes.OUTA)
def out_inst(instance: CentralProcessingUnit):
    load(instance, CPURegisterNames._HIDDEN)
    instance.DEVICE_MANAGER.write_port(instance.REGISTERS._H, instance.REGISTERS.AA)


@CentralProcessingUnit.register_instruction(InstructionCodes.ADD)
def add_inst(instance: CentralProcessingUnit):
    instance.FLAGS.carry = False
    instance.FLAGS.overflow = False
    instance.FLAGS.signed = False
    instance.FLAGS.zero = False

    sign_bit_mask = 1 << (instance.WORD_SIZE_BITS - 1)
    sign_bit_A = instance.REGISTERS.AA & sign_bit_mask
    sign_bit_B = instance.REGISTERS.AB & sign_bit_mask

    instance.REGISTERS.AA = instance.REGISTERS.AA + instance.REGISTERS.AB
    try:
        instance.REGISTERS.AA.to_bytes(length=instance.word_in_bytes)
    except OverflowError:
        # If result is too big to fit the A register set carry bit
        # Ex. 0xFFFF + 0xFFFF
        instance.FLAGS.carry = True
        instance.REGISTERS.AA = instance.REGISTERS.AA - (2**instance.WORD_SIZE_BITS)

    sign_bit_result = instance.REGISTERS.AA & sign_bit_mask

    # Is MSB of result set, aka. result could be negative if on overflow
    instance.FLAGS.signed = bool(sign_bit_result)

    # Is result exactly zero (ignoring carry and overflow)
    instance.FLAGS.zero = instance.REGISTERS.AA == 0

    if sign_bit_A == sign_bit_B and sign_bit_A != sign_bit_result:
        # If the result has incorrect sign for two's compliment numbers set overflow
        # Ex. 0x8000 (-32768) + 0xFFFF (-1)
        # Ex. 0x7fff (32767) + 0x0001 (1)
        instance.FLAGS.overflow = True


@CentralProcessingUnit.register_instruction(InstructionCodes.JMP)
def jump_to(instance: CentralProcessingUnit):
    jump(instance)


@CentralProcessingUnit.register_instruction_variants(
    [
        (InstructionCodes.JZ, {"flag_name": CPUFlagNames.zero}),
        (InstructionCodes.JO, {"flag_name": CPUFlagNames.overflow}),
        (InstructionCodes.JS, {"flag_name": CPUFlagNames.signed}),
        (InstructionCodes.JC, {"flag_name": CPUFlagNames.carry}),
    ]
)
def jump_flag(instance: CentralProcessingUnit, flag_name: CPUFlagNames):
    jump(instance, flag_name=flag_name)


@dataclass
class VideoOutputDevice:
    VRAM_OFFSET: int
    VRAM_SIZE: int
    buffer: Optional[memoryview]
    resolution: vm_types.VideoResolution
    color_format: vm_types.ColorFormats = vm_types.ColorFormats.RGB
    hardware_device_id: vm_types.HardwareDeviceIds = vm_types.HardwareDeviceIds.DISP0

    def __post_init__(self):
        if not self.buffer:
            raise ValueError("Video device missing vram")

    @property
    def VRAM(self) -> memoryview:
        if not self.buffer:
            raise ValueError("Video device missing vram")
        return self.buffer

    @virtual_machine.device_register_port(0, True)
    def get_vram_address(self):
        return self.VRAM_OFFSET

    def update_on_state_change(self, data): ...

    def device_tick(self): ...


@dataclass
class KbdInputDevice:
    key_code: int = 0
    hardware_device_id: vm_types.HardwareDeviceIds = vm_types.HardwareDeviceIds.KBD0
    buffer: Optional[memoryview] = None
    pressed_keys: set = field(default_factory=set)

    @virtual_machine.device_register_port(10)
    def is_key_pressed(self) -> int:
        return 1 if self.key_code in self.pressed_keys else 0

    @virtual_machine.device_register_port(11, False)
    def set_check_key_code(self, val):
        self.key_code = val

    def update_on_state_change(self, data):
        # TODO: How do we raise interrupts on device state change
        self.pressed_keys = data["pressed_keys"]

    def device_tick(self):
        # TODO: Add characters to keyboard buffer ?
        pass


def instance_factory() -> vm_types.GenericVirtualMachine:
    assembler_instance = Assembler(
        TEST_PROG, InstructionCodes, WORD_SIZE, INSTRUCTIONS_META, MACROS_META
    )

    SCREEN_WIDTH = 640 // 8
    SCREEN_HEIGHT = 400 // 8

    ram_size = 4 * 1024
    vram_size = SCREEN_WIDTH * SCREEN_HEIGHT * vm_types.ColorFormats.RGB.value.byte_size
    memory = bytearray(ram_size + vram_size)
    ram = memoryview(memory)
    vram = memoryview(memory)[ram_size : ram_size + vram_size]

    device_manager = virtual_machine.DeviceManager(
        video_devices={
            vm_types.HardwareDeviceIds.DISP0: VideoOutputDevice(
                ram_size,
                vram_size,
                vram,
                vm_types.VideoResolution(SCREEN_WIDTH, SCREEN_HEIGHT),
            )
        },
        char_devices={vm_types.HardwareDeviceIds.KBD0: KbdInputDevice()},
    )

    cpu_instance = CentralProcessingUnit(
        HALT_INS_CODE=HALT_INS_CODE,
        RAM=ram,
        FLAGS=CPUFlags(),
        DEVICE_MANAGER=device_manager,
    )

    vm_instance = virtual_machine.VirtualMachine(
        memory=ram,
        cpu=cpu_instance,
        assembler=assembler_instance,
        device_manager=device_manager,
    )

    vm_instance.load_program_at(0, TEST_PROG)
    vm_instance.restart()

    return vm_instance


if __name__ == "__main__":
    import log

    log.init_logging()

    vm_instance = instance_factory()

    vm_instance.run()
