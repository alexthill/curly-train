use scop_lib::vulkan::VkApp;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
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
struct App {
    window: Option<Window>,
    vulkan: Option<VkApp>,
    fps: Option<(Instant, u32)>,
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

    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {
        let Some(app) = self.vulkan.as_mut() else { return };
        app.wheel_delta = None;
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
            WindowEvent::Resized { .. } => {
                self.vulkan.as_mut().unwrap().dirty_swapchain = true;
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.vulkan.as_mut().unwrap().is_left_clicked =
                    state == ElementState::Pressed && button == MouseButton::Left;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let app = self.vulkan.as_mut().unwrap();

                let position: (i32, i32) = position.into();
                app.cursor_delta = Some([
                    app.cursor_position[0] - position.0,
                    app.cursor_position[1] - position.1,
                ]);
                app.cursor_position = [position.0, position.1];
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, v_lines),
                ..
            } => {
                self.vulkan.as_mut().unwrap().wheel_delta = Some(v_lines);
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
        app.dirty_swapchain = app.draw_frame();
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.vulkan.as_ref().unwrap().wait_gpu_idle();
    }
}
