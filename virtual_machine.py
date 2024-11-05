from dataclasses import dataclass, field
import logging
from typing import Callable, MutableMapping, Optional, cast

import vm_types

logger = logging.getLogger(__name__)


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
    video_device: Optional[vm_types.GenericVideoDevice] = None
    _WRITE_TO_PORTS: MutableMapping[int, Callable] = field(default_factory=dict)
    _READ_FROM_PORTS: MutableMapping[int, Callable] = field(default_factory=dict)

    def __post_init__(self):
        self._register_devices([self.video_device])

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
                            f"Port id={port_handler.info.port_id} " "already registerd"
                        )
                    port_mapping[port_handler.info.port_id] = port_handler

    def read_port(self, port_id: int) -> int: ...

    def write_port(self, port_id: int, value: int): ...


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

    @property
    def video_memory(self) -> bytearray | memoryview:
        # Place holder until video memory gets implemented
        if self.device_manager.video_device:
            bytes_data = self.device_manager.video_device.VRAM
        else:
            bytes_data = bytearray(320 * 200 * 3)
            for y in range(0, 200):
                for x in range(0, 320, 1):
                    row = (320 * 3) * y
                    col = x * 3
                    bytes_data[row + col] = (x ^ y) % 255
                    bytes_data[row + col + 1] = 0
                    bytes_data[row + col + 2] = 0

        return bytes_data

    @property
    def video_resolution(self):
        if self.device_manager.video_device:
            return self.device_manager.video_device.resolution
        return vm_types.VideoResolution(320, 200)

    def load_at(self, address: int, data: bytearray, force=False):
        # TODO: I think there is 1 off bug here looking how mem changes
        data_len = len(data)
        memory_len = len(self.memory)

        if not force and address % self.cpu.word_in_bytes != 0:
            raise Exception("TODO")

        if address < 0 or (address + data_len - 1) >= memory_len:
            raise Exception("TODO")

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
