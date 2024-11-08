from dataclasses import dataclass, field
from functools import partial
import logging
from typing import Callable, Mapping, MutableMapping, Optional, cast

import vm_types

logger = logging.getLogger(__name__)


@dataclass
class DefaultVideoDevice:
    hardware_device_id = vm_types.HardwareDeviceIds.DISP0
    color_format: vm_types.ColorFormats = vm_types.ColorFormats.RGB
    resolution: vm_types.VideoResolution = vm_types.VideoResolution(320, 200)
    buffer: Optional[memoryview] = None

    def __post_init__(self):
        self.buffer = memoryview(self.draw_pattern())

    @property
    def VRAM(self) -> memoryview:
        if not self.buffer:
            raise ValueError("Video device missing vram")
        return self.buffer

    def draw_pattern(self) -> bytearray:
        # Place holder if no video device
        w = self.resolution.width
        h = self.resolution.heigth
        color_depth = vm_types.ColorFormats.RGB.value.byte_size
        bytes_data = bytearray(w * h * color_depth)
        for y in range(0, h):
            for x in range(0, w, 1):
                row = (w * color_depth) * y
                col = x * color_depth
                bytes_data[row + col] = (x ^ y) % 255
                bytes_data[row + col + 1] = 0
                bytes_data[row + col + 2] = 0

        return bytes_data

    def update_on_state_change(self, data): ...

    def device_tick(self): ...


def device_register_port(
    port_id: int, is_read_from_port: bool = True
) -> vm_types.DecoratorCallable:
    def decorator(method) -> Callable:
        return vm_types.PortLabeldCallable(
            method, vm_types.PortInfo(port_id, is_read_from_port)
        )

    return decorator


@dataclass
class DeviceManager:
    video_devices: MutableMapping[
        vm_types.HardwareDeviceIds, vm_types.GenericVideoDevice
    ] = field(default_factory=dict)
    char_devices: Mapping[
        vm_types.HardwareDeviceIds, vm_types.GenericCharacterDevice
    ] = field(default_factory=dict)
    _WRITE_TO_PORTS: MutableMapping[int, Callable] = field(default_factory=dict)
    _READ_FROM_PORTS: MutableMapping[int, Callable] = field(default_factory=dict)

    def __post_init__(self):
        if vm_types.HardwareDeviceIds.DISP0 not in self.video_devices:
            self.video_devices.update(
                {vm_types.HardwareDeviceIds.DISP0: DefaultVideoDevice()}
            )

        self.all_devices = list(self.video_devices.values()) + list(
            self.char_devices.values()
        )

        self._register_devices(self.all_devices)

    def _register_devices(self, devices):
        for device_object in devices:
            for attr_name in dir(device_object):
                attr = getattr(device_object, attr_name)
                if callable(attr) and hasattr(
                    attr, vm_types.PortLabeldCallable.IS_PORT_HANDLER
                ):
                    port_mapping = self._WRITE_TO_PORTS
                    if (
                        port_handler := cast(vm_types.PortLabeldCallable, attr)
                    ).info.read_port:
                        port_mapping = self._READ_FROM_PORTS
                    if port_handler.info.port_id in port_mapping:
                        raise ValueError(
                            f"Port id={port_handler.info.port_id} " "already registered"
                        )
                    port_mapping[port_handler.info.port_id] = partial(
                        port_handler, device_object
                    )

    def process_devices_on_tick(self):
        for device in self.all_devices:
            device.device_tick()

    def error_port_not_registered(self, port_id: int, read_port=True):
        port_mapping = self._WRITE_TO_PORTS
        if read_port:
            port_mapping = self._READ_FROM_PORTS

        if port_id not in port_mapping:
            raise ValueError(f"Port id={port_id} {read_port=} not registered")

    def read_port(self, port_id: int) -> int:
        self.error_port_not_registered(port_id)
        return self._READ_FROM_PORTS[port_id]()

    def write_port(self, port_id: int, value: int):
        self.error_port_not_registered(port_id, read_port=False)
        self._WRITE_TO_PORTS[port_id](value)

    def get_video_device(
        self, hardware_device_id
    ) -> Optional[vm_types.GenericVideoDevice]:
        # TODO: (Hristo) this should be structured/ typed in a better way
        video_device = self.video_devices.get(hardware_device_id)
        return video_device

    def get_char_device(
        self, hardware_device_id
    ) -> Optional[vm_types.GenericCharacterDevice]:
        # TODO: (Hristo) this should be structured/ typed in a better way
        char_device = self.char_devices.get(hardware_device_id)
        return char_device


@dataclass
class VirtualMachine:
    memory: memoryview
    cpu: (
        vm_types.GenericCentralProcessingUnit
        | vm_types.GenericCentralProcessingUnitDevicePorts
    )
    assembler: vm_types.GenericAssembler
    device_manager: vm_types.GenericDeviceManager = field(default_factory=DeviceManager)
    stack_address: int | None = None
    clock_speed_hz: int = 1
    _ellapsed_since_last: float = 0

    def __post_init__(self):
        self.restart()

    def _replace_memory(self, memory: bytearray):
        self.memory[:] = memory

    def get_registers(self):
        return self.cpu.dump_registers()

    def get_program_text(self):
        return self.assembler.text

    def get_current_instruction_address(self):
        return self.cpu.current_inst_address

    def load_at(self, address: int, data: bytearray, force=False):
        data_len = len(data)
        memory_len = len(self.memory)

        if not force and address % self.cpu.word_in_bytes != 0:
            raise ValueError(
                f"Address {hex(address)} in not word boundary "
                "use `force=True` to load anyway"
            )

        if address < 0 or (address + data_len - 1) >= memory_len:
            raise ValueError(
                f"Address {hex(address)} out side of range " f" 0 to {hex(memory_len)}"
            )

        self.memory[address : len(data)] = data

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
        self.device_manager.process_devices_on_tick()
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
