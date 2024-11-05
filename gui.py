from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Mapping, NamedTuple

import pygame
import pygame_gui

import vm_types

logger = logging.getLogger(__name__)


@dataclass
class GUIText:
    title: str = "🐍 Snake toy VM"
    display_title: str = "Display"
    memory_title: str = "Memory"
    registers_title: str = "Registers"
    program_title: str = "Program Listing"
    controls_title: str = "Control panel"
    start_button_cta: str = "Stopped!"
    stop_button_cta: str = "Running..."


class Resolution(NamedTuple):
    w: int
    h: int


class EventHandlingComponent:
    def register_eventable_component(self, component):
        if not self.eventable_components:
            self.eventable_components = set()

        self.eventable_components.add(component)


class Controls(EventHandlingComponent):
    def __init__(
        self,
        dim: pygame.Rect,
        manager,
        title,
        stoped_cta: str,
        running_cta: str,
        available_vms: Mapping[str, vm_types.GenericVirtualMachine],
        ui_root,
    ) -> None:
        self.ui_root = ui_root
        self.eventable_components = None
        self.stoped_cta = stoped_cta
        self.running_cta = running_cta

        controls_window = pygame_gui.elements.UIWindow(
            element_id="controls_window",
            window_display_title=title,
            rect=dim,
            manager=manager,
        )
        inner_window_size = controls_window.get_container().get_size()

        dropdown_dim = pygame.Rect((0, 0), (inner_window_size[0], 25))
        vm_options = list(available_vms.keys())
        self.vm_select_dropdown = pygame_gui.elements.UIDropDownMenu(
            object_id="vm_select_dropdown",
            options_list=vm_options,
            starting_option=vm_options[0],
            relative_rect=dropdown_dim,
            container=controls_window,
            manager=manager,
        )

        self.register_eventable_component(self.vm_select_dropdown)

        toggle_run_dim = pygame.Rect(
            (0, dropdown_dim.h), (dropdown_dim.w, dropdown_dim.h)
        )

        toggle_run_button = pygame_gui.elements.UIButton(
            object_id="toggle_run_button",
            relative_rect=toggle_run_dim,
            text=self.stoped_cta,
            container=controls_window,
            manager=manager,
        )

        self.register_eventable_component(toggle_run_button)

        program_path_dim = pygame.Rect(
            (0, dropdown_dim.h * 2), (dropdown_dim.w, dropdown_dim.h)
        )

        self.program_path_input = pygame_gui.elements.UITextEntryLine(
            object_id="program_path_input",
            relative_rect=program_path_dim,
            container=controls_window,
            manager=manager,
        )

        load_program_dim = pygame.Rect(
            (0, dropdown_dim.h * 3), (dropdown_dim.w, dropdown_dim.h)
        )

        load_program_button = pygame_gui.elements.UIButton(
            object_id="load_program_button",
            relative_rect=load_program_dim,
            text="Load program",
            container=controls_window,
            manager=manager,
        )

        self.register_eventable_component(load_program_button)

        restart_dim = pygame.Rect(
            (0, dropdown_dim.h * 4), (dropdown_dim.w // 3, dropdown_dim.h)
        )

        restart_button = pygame_gui.elements.UIButton(
            object_id="restart_button",
            relative_rect=restart_dim,
            text="Restart",
            container=controls_window,
            manager=manager,
        )

        self.register_eventable_component(restart_button)

        reset_dim = pygame.Rect(
            (dropdown_dim.w // 3, dropdown_dim.h * 4),
            (dropdown_dim.w // 3, dropdown_dim.h),
        )

        reset_button = pygame_gui.elements.UIButton(
            object_id="reset_button",
            relative_rect=reset_dim,
            text="Reset",
            container=controls_window,
            manager=manager,
        )

        self.register_eventable_component(reset_button)

    def refresh(self): ...

    def handle_events(self, event):
        if event.ui_element not in self.eventable_components:
            return

        event_object_ids = set(event.ui_element.get_object_ids())
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if "vm_select_dropdown" in event_object_ids:
                vm_key = event.text
                self.ui_root.swap_vm(vm_key)

        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if "toggle_run_button" in event_object_ids:
                self.ui_root.vm_is_running = not self.ui_root.vm_is_running
                if self.ui_root.vm_is_running:
                    cta = self.running_cta
                else:
                    cta = self.stoped_cta
                event.ui_element.set_text(cta)
            elif "load_program_button" in event_object_ids:
                program_path = self.program_path_input.get_text()
                try:
                    with open(program_path, "r") as f:
                        program_text = f.read()
                except Exception as e:
                    program_text = str(e)

                self.ui_root.load_program(program_text)
            elif "reset_button" in event_object_ids:
                self.ui_root.restart(full_reset=True)
            elif "restart_button" in event_object_ids:
                self.ui_root.restart()


class Dispaly:
    def __init__(
        self,
        dim: pygame.Rect,
        vm_display_rect: pygame.Rect,
        manager,
        title,
        ui_root,
    ) -> None:
        self.ui_root = ui_root

        display_window = pygame_gui.elements.UIWindow(
            element_id="display_window",
            window_display_title=title,
            rect=vm_display_rect,
            manager=manager,
        )

        self.display_surface_dims = Resolution(dim.w, dim.h)
        display_surface = pygame.Surface(self.display_surface_dims)
        self.display_image = pygame_gui.elements.UIImage(
            relative_rect=dim,
            container=display_window,
            image_surface=display_surface,
            manager=manager,
        )

    def refresh(self):
        image_bytes = self.ui_root.vm.video_memory
        resolution = self.ui_root.vm.video_resolution
        source = pygame.image.frombuffer(
            image_bytes,
            # (self.display_surface_dims.w // 2, self.display_surface_dims.h // 2),
            resolution,
            "RGB",
        )
        self.display_image.set_image(source)


class Program:
    def __init__(self, dim: pygame.Rect, manager, title, ui_root) -> None:
        self.ui_root = ui_root
        memory_window = pygame_gui.elements.UIWindow(
            element_id="program_window",
            window_display_title=title,
            rect=dim,
            manager=manager,
        )
        window_dim = pygame.Rect((0, 0), memory_window.get_container().get_size())
        self.text_box = pygame_gui.elements.UITextBox(
            html_text=self.ui_root.vm.get_program_text(),
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

    def refresh(self):
        if not self.ui_root.state_changed:
            return

        self.text_box.set_text(html_text=self.ui_root.vm.get_program_text())


class Memory:
    def __init__(self, dim: pygame.Rect, manager, title, ui_root) -> None:
        self.ui_root = ui_root
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
        mem_line_count = math.ceil(len(ui_root.vm.memory) / 16)

        for _ in range(mem_line_count):
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((0, position), (parent_width, -1)),
                container=self.text_box,
                text="[00000000] 0000 . 0000 . 0000 .0000 . 0000 . 0000 . 0000 . 0000",
                manager=manager,
            )

            self.lines.append(label)
            position += 13

    def refresh(self):
        memory_len = len(self.ui_root.vm.memory)
        offset = 0
        word_size = 2
        words = deque()
        instruction_counter = self.ui_root.vm.get_current_instruction_address()
        while offset < memory_len:
            hex_str = self.ui_root.vm.memory[offset : offset + word_size].hex()
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
    def __init__(self, dim: pygame.Rect, manager, title, ui_root) -> None:
        self.ui_root = ui_root

        registers_window = pygame_gui.elements.UIWindow(
            element_id="registers_window",
            window_display_title=title,
            rect=dim,
            manager=manager,
        )
        registers_data = self.ui_root.vm.get_registers()
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

    def refresh(self):
        reg_data = self.ui_root.vm.get_registers()

        for label, (reg_name, reg_val) in zip(self.registers, reg_data.items()):
            reg_text = f"{reg_name}={reg_val.to_bytes(length=2).hex()}"
            label.set_text(reg_text)


TITLE_BAR_HEIGHT = 28


class VirualMachineGUI:
    def __init__(
        self,
        available_vms: Mapping[str, vm_types.GenericVirtualMachine],
        resolution=Resolution(1024, 720),
        vm_resolution=Resolution(640, 400),
        text_data: GUIText = GUIText(),
    ) -> None:
        pygame.init()
        pygame.display.set_caption(text_data.title)

        self.is_running = True
        self.vm_is_running = False
        self.state_changed = True
        self.available_vms = available_vms
        self.vm = list(available_vms.values())[0]

        self.clock = pygame.time.Clock()
        self.window_surface = pygame.display.set_mode(resolution)
        self.background = pygame.Surface(resolution)
        self.background.fill(pygame.Color("#000000"))
        self.manager = pygame_gui.UIManager(resolution)

        vm_window_dim = Resolution(vm_resolution.w, vm_resolution.h + TITLE_BAR_HEIGHT)
        vm_display_rect = pygame.Rect((0, 0), vm_window_dim)
        vm_display_image_rect = pygame.Rect((0, 0), vm_resolution)

        vm_memory_rect = pygame.Rect(
            (0, vm_window_dim.h), (vm_window_dim.w, resolution.h - vm_window_dim.h)
        )

        vm_controls_rect = pygame.Rect(
            (vm_window_dim.w, 0), (resolution.w - vm_window_dim.w, resolution.h // 4)
        )

        vm_registers_rect = pygame.Rect(
            (vm_window_dim.w, resolution.h // 4),
            (resolution.w - vm_window_dim.w, resolution.h // 4),
        )

        vm_program_rect = pygame.Rect(
            (vm_window_dim.w, resolution.h // 2),
            (resolution.w - vm_window_dim.w, resolution.h // 2),
        )

        self.display = Dispaly(
            vm_display_image_rect,
            vm_display_rect,
            self.manager,
            text_data.display_title,
            ui_root=self,
        )

        self.mem = Memory(
            vm_memory_rect, self.manager, text_data.memory_title, ui_root=self
        )

        self.controls = Controls(
            vm_controls_rect,
            self.manager,
            text_data.controls_title,
            text_data.start_button_cta,
            text_data.stop_button_cta,
            available_vms=available_vms,
            ui_root=self,
        )

        self.regs = Registers(
            vm_registers_rect, self.manager, text_data.registers_title, ui_root=self
        )

        self.program = Program(
            vm_program_rect, self.manager, text_data.program_title, ui_root=self
        )

        self.eventable_components = [self.controls]

    def swap_vm(self, vm_key):
        self.vm = self.available_vms[vm_key]
        self.state_changed = True

    def load_program(self, program_text):
        self.vm.load_and_reset(program_text)
        self.state_changed = True

    def restart(self, full_reset=False):
        if full_reset:
            self.vm.reset()
        else:
            self.vm.restart()
        self.state_changed = True

    def refresh(self):
        self.mem.refresh()
        self.regs.refresh()
        self.display.refresh()
        self.program.refresh()
        self.state_changed = False

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

                if hasattr(event, "ui_element"):
                    for component in self.eventable_components:
                        component.handle_events(event)

                self.manager.process_events(event)

            if self.vm_is_running:
                self.vm.step(time_delta_milli)

            self.refresh()

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
