from dataclasses import dataclass
import logging
from typing import Mapping, NamedTuple

import pygame
import pygame_gui

import vm_types

logger = logging.getLogger(__name__)


START_SPEED_HZ = 1


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
            (0, dropdown_dim.h), (dropdown_dim.w // 3, dropdown_dim.h)
        )
        toggle_run_button = pygame_gui.elements.UIButton(
            object_id="toggle_run_button",
            relative_rect=toggle_run_dim,
            text=self.stoped_cta,
            container=controls_window,
            manager=manager,
        )
        self.register_eventable_component(toggle_run_button)

        restart_dim = pygame.Rect(
            (dropdown_dim.w // 3, dropdown_dim.h), (dropdown_dim.w // 3, dropdown_dim.h)
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
            ((dropdown_dim.w // 3) * 2, dropdown_dim.h),
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

        mhz_label_dim = pygame.Rect(
            (0, dropdown_dim.h * 2), (dropdown_dim.w // 2, dropdown_dim.h)
        )
        self.mhz_label = pygame_gui.elements.UILabel(
            relative_rect=mhz_label_dim,
            container=controls_window,
            text=str(START_SPEED_HZ),
            manager=manager,
        )

        ins_per_tick_label_dim = pygame.Rect(
            (dropdown_dim.w // 2, dropdown_dim.h * 2),
            (dropdown_dim.w // 2, dropdown_dim.h),
        )
        self.ins_per_tick_label = pygame_gui.elements.UILabel(
            relative_rect=ins_per_tick_label_dim,
            container=controls_window,
            text="0",
            manager=manager,
        )

        mhz_slider_dim = pygame.Rect(
            (0, dropdown_dim.h * 3), (dropdown_dim.w, dropdown_dim.h)
        )
        mhz_slider = pygame_gui.elements.UI2DSlider(
            relative_rect=mhz_slider_dim,
            start_value_x=START_SPEED_HZ,
            start_value_y=0,
            value_range_x=(1, 1000),
            value_range_y=(0, 0),
            container=controls_window,
            manager=manager,
        )
        mhz_slider.set_current_value(1, 0, False)
        self.register_eventable_component(mhz_slider)

        program_path_dim = pygame.Rect(
            (0, dropdown_dim.h * 4), (dropdown_dim.w, dropdown_dim.h)
        )

        self.program_path_input = pygame_gui.elements.UITextEntryLine(
            object_id="program_path_input",
            relative_rect=program_path_dim,
            container=controls_window,
            manager=manager,
        )

        load_program_dim = pygame.Rect(
            (0, dropdown_dim.h * 5), (dropdown_dim.w, dropdown_dim.h)
        )

        load_program_button = pygame_gui.elements.UIButton(
            object_id="load_program_button",
            relative_rect=load_program_dim,
            text="Load program",
            container=controls_window,
            manager=manager,
        )

        self.register_eventable_component(load_program_button)

    def refresh(self):
        self.ins_per_tick_label.set_text(str(self.ui_root.vm.instructions_per_tick))

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
        elif event.type == pygame_gui.UI_2D_SLIDER_MOVED:
            hertz, _ = event.value
            self.ui_root.vm.set_clock_speed(hertz)
            self.mhz_label.set_text(str(hertz))
            logger.debug(f"Setting speed to {hertz=}")


class Dispaly(EventHandlingComponent):
    def __init__(
        self,
        dim: pygame.Rect,
        vm_display_rect: pygame.Rect,
        manager,
        title,
        ui_root,
    ) -> None:
        self.ui_root = ui_root

        self.manager = manager

        self.pressed_keys = set()

        self.display_window = pygame_gui.elements.UIWindow(
            element_id="display_window",
            window_display_title=title,
            rect=vm_display_rect,
            manager=manager,
        )

        self.display_surface_dims = Resolution(dim.w, dim.h)
        display_surface = pygame.Surface(self.display_surface_dims)
        self.display_image = pygame_gui.elements.UIImage(
            relative_rect=dim,
            container=self.display_window,
            image_surface=display_surface,
            manager=manager,
        )

    def refresh(self):
        video_device = self.ui_root.vm.device_manager.get_video_device(
            hardware_device_id=vm_types.HardwareDeviceIds.DISP0
        )
        image_bytes = video_device.VRAM
        resolution = video_device.resolution
        color_format = video_device.color_format.value.code
        source = pygame.image.frombuffer(
            image_bytes,
            resolution,
            color_format,
        )
        self.display_image.set_image(source)

    def handle_events(self, event):
        if self.display_window in (self.manager.get_focus_set() or {}):
            if event.type == pygame.KEYDOWN:
                self.pressed_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self.pressed_keys.remove(event.key)

            self.ui_root.vm.device_manager.get_char_device(
                hardware_device_id=vm_types.HardwareDeviceIds.KBD0
            ).update_on_state_change(data={"pressed_keys": self.pressed_keys})


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
        self.memory_root = memory_window
        self.manager = manager
        # TODO: (Hristo) Make arch agnostic
        self.word_size = 2
        self.words_in_row = 8
        self.rows_in_page = 16
        self.page_size_bytes = self.rows_in_page * self.words_in_row * self.word_size
        self.memeory_size = len(self.ui_root.vm.memory)
        self.current_row = 0
        self.padding_top = 5
        self.padding_left = 5
        self.address_w = 160
        self.word_w = 50
        self.spacer_w = 5
        self.spacer_h = 3
        window_dim = pygame.Rect((0, 0), memory_window.get_container().get_size())
        self.scroll_bar_width = 20
        self.page_width = window_dim.w - self.scroll_bar_width - self.padding_left
        self.row_height = 12
        scroll_bar_dim = pygame.Rect(
            (self.page_width + 1, 0),
            (self.scroll_bar_width, memory_window.get_container().get_size()[1]),
        )

        self.page_elements = self._make_page()

        self.scroll_bar = pygame_gui.elements.UIVerticalScrollBar(
            relative_rect=scroll_bar_dim,
            container=memory_window,
            visible_percentage=0.5,
        )

    def _make_page(
        self, dummy_address_data="[00000000]", dummy_word_filler=b"\xff\xff"
    ):
        dummy_word_data = [dummy_word_filler for _ in range(self.words_in_row)]
        page = pygame_gui.core.UIContainer(
            relative_rect=pygame.Rect(
                (self.padding_left, self.padding_top),
                (
                    self.page_width,
                    self.memory_root.get_container().get_size()[1] - self.padding_top,
                ),
            ),
            container=self.memory_root,
            manager=self.manager,
        )

        self._make_row(
            page,
            0,
            address_data="ADDRESS",
            word_data=[
                f"+{hex(offset)}"
                for offset in range(
                    0, self.words_in_row * self.word_size, self.word_size
                )
            ],
        )
        y_offset = self.row_height + self.spacer_h
        data = [d.hex() for d in dummy_word_data]

        page_rows = []
        for _ in range(self.rows_in_page):
            row_elements = self._make_row(
                page, y_offset, address_data=dummy_address_data, word_data=data
            )
            y_offset += self.row_height + self.spacer_h
            page_rows.append(row_elements)
        return page_rows

    def _make_row(self, parent, y_offset, address_data, word_data):
        row = pygame_gui.core.UIContainer(
            relative_rect=pygame.Rect((0, y_offset), parent.get_container().get_size()),
            container=parent,
            manager=self.manager,
        )

        address_element = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 0), (self.address_w, -1)),
            container=row,
            text=address_data.upper(),
            manager=self.manager,
        )

        offset = self.address_w + self.spacer_w
        row_word_elements = []
        for data in word_data:
            word_element = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((offset, 0), (self.word_w, -1)),
                container=row,
                text=data.upper(),
                manager=self.manager,
            )
            offset += self.word_w + self.spacer_w
            row_word_elements.append(word_element)

        return (address_element, row_word_elements)

    def _update_page(self, mem_offset, mem_bytes):
        instruction_counter = self.ui_root.vm.get_current_instruction_address()
        for row_idx, (row_address_element, row_data_elements) in enumerate(
            self.page_elements
        ):
            row_offset = row_idx * self.words_in_row * self.word_size
            address_text = f"[{(mem_offset + row_offset).to_bytes(length=8).hex()}]"
            row_address_element.set_text(address_text.upper())
            for word_idx, data_element in enumerate(row_data_elements):
                word_offset = row_offset + word_idx * self.word_size
                word_text = mem_bytes[word_offset : word_offset + self.word_size].hex()
                if word_offset + mem_offset == instruction_counter:
                    word_text = f"󰯉 {word_text}"

                data_element.set_text(word_text.upper())

    def refresh(self):
        bytes_in_row = self.words_in_row * self.word_size
        if self.ui_root.state_changed:
            self.memeory_size = len(self.ui_root.vm.memory)
            visible_percentage = self.page_size_bytes / self.memeory_size
            self.current_row = 0
            self.scroll_bar.set_visible_percentage(visible_percentage)
            self.scroll_bar.set_scroll_from_start_percentage(0)

        elif self.scroll_bar.check_has_moved_recently():
            self.current_row = int(
                (self.memeory_size // bytes_in_row) * self.scroll_bar.start_percentage
            )

        page_offset = self.current_row * bytes_in_row
        mem_data = self.ui_root.vm.memory[
            page_offset : page_offset + self.page_size_bytes
        ]
        self._update_page(page_offset, mem_data)


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
        resolution=Resolution(1224, 720),
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

        self._hack_in_default_font()

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
        self.input_eventable_components = [self.display]

    def _hack_in_default_font(self):
        """
        Not a fan at all of how theming is setup in `pygame-gui` esp. if just
        setting one default font across app. So we are just going to hack that in.
        Should have picked a diff. UI lib. for other reasons as well but,
        good enough.
        *shrugs*
        """
        font_size = 12
        font_name = "blue_term"
        self.manager.ui_theme.font_dict._load_single_font_style(  # type: ignore
            ("assests/BigBlueTermPlusNerdFontMono-Regular.ttf", False),
            font_name,
            font_size,
            font_style={  # type: ignore
                "bold": False,
                "italic": False,
                "antialiased": False,
                "script": "Latn",
                "direction": pygame.DIRECTION_LTR,
            },
            force_immediate_load=True,
        )
        font_id = self.manager.ui_theme.font_dict.create_font_id(  # type: ignore
            font_size, font_name, False, False, True
        )  # type: ignore
        default_font = pygame_gui.core.ui_font_dictionary.DefaultFontData(  # type: ignore
            font_size,
            font_name,
            "regular",
            "assests/BigBlueTermPlusNerdFontMono-Regular.ttf",
            "assests/BigBlueTermPlusNerdFontMono-Regular.ttf",
            "assests/BigBlueTermPlusNerdFontMono-Regular.ttf",
            "assests/BigBlueTermPlusNerdFontMono-Regular.ttf",
        )

        self.manager.ui_theme.font_dict.loaded_fonts[font_id] = (  # type: ignore
            self.manager.ui_theme.font_dict.loaded_fonts[font_name]  # type: ignore
        )
        self.manager.ui_theme.font_dict.default_font_dictionary["en"] = default_font  # type: ignore
        self.manager.ui_theme.font_dict.default_font = default_font  # type: ignore

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
        self.controls.refresh()
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
                elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                    for component in self.input_eventable_components:
                        component.handle_events(event)

                self.manager.process_events(event)

            if self.vm_is_running:
                self.vm.step(time_delta_milli)

            self.refresh()

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
