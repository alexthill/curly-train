use scop_lib::vulkan::VkApp;

use cgmath::{Matrix4, Vector3};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Window, WindowId},
};
use std::time::Instant;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const TITLE: &str = "scop";

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
pub struct KeyStates {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan: Option<VkApp>,

    fps: Option<(Instant, u32)>,
    last_frame: Option<Instant>,

    pressed: KeyStates,
    toggle_rotate: bool,
    is_left_clicked: bool,
    cursor_position: Option<[i32; 2]>,
    cursor_delta: [i32; 2],
    wheel_delta: f32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(TITLE)
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.vulkan = Some(VkApp::new(&window, WIDTH, HEIGHT));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(
                            key @ (
                                KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD
                                | KeyCode::Space | KeyCode::ShiftLeft
                                | KeyCode::KeyR
                            )
                        ),
                        repeat: false,
                        ..
                    },
                ..
            } => {
                match key {
                    KeyCode::KeyW => self.pressed.forward = state.is_pressed(),
                    KeyCode::KeyA => self.pressed.left = state.is_pressed(),
                    KeyCode::KeyS => self.pressed.backward = state.is_pressed(),
                    KeyCode::KeyD => self.pressed.right = state.is_pressed(),
                    KeyCode::Space => self.pressed.up = state.is_pressed(),
                    KeyCode::ShiftLeft => self.pressed.down = state.is_pressed(),
                    KeyCode::KeyR if state.is_pressed() => self.toggle_rotate = !self.toggle_rotate,
                    _ => {}
                }
            }
            WindowEvent::Resized { .. } => {
                self.vulkan.as_mut().unwrap().dirty_swapchain = true;
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.is_left_clicked =
                    state == ElementState::Pressed && button == MouseButton::Left;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos: (i32, i32) = position.into();
                if self.is_left_clicked {
                    if let Some(old_pos) = self.cursor_position {
                        self.cursor_delta[0] += new_pos.0 - old_pos[0];
                        self.cursor_delta[1] += new_pos.1 - old_pos[1];
                    }
                }
                self.cursor_position = Some([new_pos.0, new_pos.1]);
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, v_lines),
                ..
            } => {
                self.wheel_delta = v_lines;
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        use std::io::Write;

        if let Some((start, count)) = self.fps.as_mut() {
            let time = start.elapsed();
            *count += 1;
            if time.as_millis() > 1000 {
                eprint!("fps: {}        \r", *count as f32 / time.as_secs_f32());
                std::io::stdout().flush().unwrap();
                *start = Instant::now();
                *count = 0;
            }
        } else {
            self.fps = Some((Instant::now(), 0));
        }

        let app = self.vulkan.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();

        if app.dirty_swapchain {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                app.recreate_swapchain();
            } else {
                return;
            }
        }

        let elapsed = self.last_frame.map(|instant| instant.elapsed()).unwrap_or_default();
        let delta = elapsed.as_secs_f32();
        self.last_frame = Some(Instant::now());

        let mut translation = Vector3::new(0., 0., 0.);
        if self.pressed.forward { translation.z += 1. * delta };
        if self.pressed.backward { translation.z -= 1. * delta };
        if self.pressed.left { translation.x += 1. * delta };
        if self.pressed.right { translation.x -= 1. * delta };
        if self.pressed.up { translation.y += 1. * delta };
        if self.pressed.down { translation.y -= 1. * delta };
        app.view_matrix = Matrix4::from_translation(translation) * app.view_matrix;

        let extent = app.get_extent();
        let x_ratio = self.cursor_delta[0] as f32 / extent.width as f32;
        let y_ratio = self.cursor_delta[1] as f32 / extent.height as f32;
        app.model_matrix = Matrix4::from_angle_y(cgmath::Deg(x_ratio * 180.)) * app.model_matrix;
        app.model_matrix = Matrix4::from_angle_x(cgmath::Deg(y_ratio * 180.)) * app.model_matrix;
        if self.toggle_rotate {
            app.model_matrix = Matrix4::from_angle_y(cgmath::Deg(delta * 90.)) * app.model_matrix;
        }
        self.cursor_delta = [0, 0];

        app.model_matrix = Matrix4::from_scale(1. + self.wheel_delta * 0.3) * app.model_matrix;
        self.wheel_delta = 0.;

        app.dirty_swapchain = app.draw_frame();
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.vulkan.as_ref().unwrap().wait_gpu_idle();
    }
}
