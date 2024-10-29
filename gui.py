from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import NamedTuple

import pygame
import pygame_gui

import toy_vm

logger = logging.getLogger(__name__)


@dataclass
class GUIText:
    title: str = "🐍 Snake toy VM"
    display_title: str = "Display"
    memory_title: str = "Memory"
    registers_title: str = "Registers"
    program_title: str = "Program Listing"


class Resolution(NamedTuple):
    w: int
    h: int


class Dispaly:
    def __init__(
        self,
        vm: toy_vm.VirtualMachine,
        dim: pygame.Rect,
        vm_display_rect: pygame.Rect,
        manager,
        title,
    ) -> None:
        self.vm = vm

        display_window = pygame_gui.elements.UIWindow(
            element_id="display_window",
            window_display_title=title,
            rect=vm_display_rect,
            manager=manager,
        )

        self.display_surface_dims = (dim.w, dim.h)
        display_surface = pygame.Surface(self.display_surface_dims)
        self.display_image = pygame_gui.elements.UIImage(
            relative_rect=dim,
            container=display_window,
            image_surface=display_surface,
            manager=manager,
        )

    def render(self):
        image_bytes = self.vm.get_video_memory()
        source = pygame.image.frombuffer(image_bytes, self.display_surface_dims, "RGB")
        self.display_image.set_image(source)


class Program:
    def __init__(
        self, vm: toy_vm.VirtualMachine, dim: pygame.Rect, manager, title
    ) -> None:
        self.vm = vm
        memory_window = pygame_gui.elements.UIWindow(
            element_id="program_window",
            window_display_title=title,
            rect=dim,
            manager=manager,
        )
        window_dim = pygame.Rect((0, 0), memory_window.get_container().get_size())
        self.text_box = pygame_gui.elements.UITextBox(
            html_text=vm.get_program_text(),
            relative_rect=window_dim,
            container=memory_window,
            manager=manager,
            anchors={
                "left": "left",
                "right": "right",
                "top": "top",
                "bottom": "bottom",
            },
        )


class Memory:
    def __init__(
        self, vm: toy_vm.VirtualMachine, dim: pygame.Rect, manager, title
    ) -> None:
        self.vm = vm
        memory_window = pygame_gui.elements.UIWindow(
            element_id="memory_window",
            window_display_title=title,
            rect=dim,
            # resizable=True,
            manager=manager,
        )
        window_dim = pygame.Rect((0, 0), memory_window.get_container().get_size())

        self.text_box = pygame_gui.elements.UIScrollingContainer(
            relative_rect=window_dim,
            allow_scroll_x=False,
            container=memory_window,
            manager=manager,
            anchors={
                "left": "left",
                "right": "right",
                "top": "top",
                "bottom": "bottom",
            },
        )

        self.lines = []

        position = 0
        parent_width = self.text_box.get_container().get_size()[0]
        mem_line_count = math.ceil(len(vm.memory) / 16)

        for _ in range(mem_line_count):
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((0, position), (parent_width, -1)),
                container=self.text_box,
                text="[00000000] 0000 . 0000 . 0000 .0000 . 0000 . 0000 . 0000 . 0000",
                manager=manager,
            )

            self.lines.append(label)
            position += 13

    def render(self):
        memory_len = len(self.vm.memory)
        offset = 0
        word_size = 2
        words = deque()
        instruction_counter = self.vm.get_current_instruction_address()
        while offset < memory_len:
            hex_str = self.vm.memory[offset : offset + word_size].hex()
            if offset == instruction_counter:
                hex_str = f"<{hex_str}>"
            words.append(hex_str)
            offset += word_size

        memory_text_lines = []
        line_no = 0
        while words:
            line_words = []
            for _ in range(min(8, len(words))):
                line_words.append(words.popleft())

            address = (line_no * 16 * 8).to_bytes(length=8).hex()
            line_text = f"[{address}] {" . ".join(line_words)}"
            memory_text_lines.append(line_text)
            line_no += 1

        for label, text in zip(self.lines, memory_text_lines):
            label.set_text(text)


class Registers:
    def __init__(
        self, vm: toy_vm.VirtualMachine, dim: pygame.Rect, manager, title
    ) -> None:
        self.vm = vm

        registers_window = pygame_gui.elements.UIWindow(
            element_id="registers_window",
            window_display_title=title,
            rect=dim,
            manager=manager,
        )
        registers_data = vm.get_registers()
        self.registers = []

        position = pygame.Vector2(0, 0)
        for idx, (reg_name, reg_val) in enumerate(registers_data.items()):
            reg_text = f"{reg_name}={reg_val.to_bytes(length=2).hex()}"
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(position, (75, -1)),
                container=registers_window,
                text=reg_text,
                manager=manager,
            )

            self.registers.append(label)
            position.x += 75
            if not (idx + 1) % 4:
                position.y += 13
                position.x = 0

    def render(self):
        reg_data = self.vm.get_registers()

        for label, (reg_name, reg_val) in zip(self.registers, reg_data.items()):
            reg_text = f"{reg_name}={reg_val.to_bytes(length=2).hex()}"
            label.set_text(reg_text)


TITLE_BAR_HEIGHT = 28


class VirualMachineGUI:
    def __init__(
        self,
        virtual_machine: toy_vm.VirtualMachine,
        resolution=Resolution(1024, 720),
        vm_resolution=Resolution(640, 400),
        text_data: GUIText = GUIText(),
    ) -> None:
        vm_window_dim = Resolution(vm_resolution.w, vm_resolution.h + TITLE_BAR_HEIGHT)
        vm_display_rect = pygame.Rect((0, 0), vm_window_dim)
        vm_display_image_rect = pygame.Rect((0, 0), vm_resolution)

        vm_memory_rect = pygame.Rect(
            (0, vm_window_dim.h), (vm_window_dim.w, resolution.h - vm_window_dim.h)
        )

        vm_registers_rect = pygame.Rect(
            (vm_window_dim.w, 0), (resolution.w - vm_window_dim.w, resolution.h // 2)
        )

        vm_program_rect = pygame.Rect(
            (vm_window_dim.w, resolution.h // 2),
            (resolution.w - vm_window_dim.w, resolution.h // 2),
        )

        pygame.init()
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.vm = virtual_machine

        pygame.display.set_caption(text_data.title)
        self.window_surface = pygame.display.set_mode(resolution)

        self.background = pygame.Surface(resolution)
        self.background.fill(pygame.Color("#000000"))

        self.manager = pygame_gui.UIManager(resolution)

        self.display = Dispaly(
            virtual_machine,
            vm_display_image_rect,
            vm_display_rect,
            self.manager,
            text_data.display_title,
        )

        self.mem = Memory(
            virtual_machine, vm_memory_rect, self.manager, text_data.memory_title
        )

        self.regs = Registers(
            virtual_machine, vm_registers_rect, self.manager, text_data.registers_title
        )

        self.program = Program(
            virtual_machine, vm_program_rect, self.manager, text_data.program_title
        )

    def render_components(self):
        self.mem.render()
        self.regs.render()
        self.display.render()

    def run(self):
        while self.is_running:
            time_delta_milli = self.clock.tick(60)
            time_delta = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                        continue

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    pass

                self.manager.process_events(event)

            self.vm.step(time_delta_milli)

            self.render_components()

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
