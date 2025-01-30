use scop_lib::fs::{self, Carousel};
use scop_lib::math::{Deg, Matrix4, Vector3};
use scop_lib::obj::NormalizedObj;
use scop_lib::vulkan::{ShaderSpv, VkApp};

use anyhow::Context;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};
use std::path::Path;
use std::time::Instant;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const TITLE: &str = "scop";

fn check_if_obj(path: &Path) -> bool {
    path.extension().map(|ext| ext == "obj").unwrap_or_default()
}

fn check_if_image(path: &Path) -> bool {
    path.extension().map(|ext| ext == "jpg" || ext == "png").unwrap_or_default()
}

fn main() {
    println!("Usage:");
    println!("Run with RUST_LOG=debug to see logging output");
    println!();
    println!("Left-Click: rotate model with mouse");
    println!("Mouse-Wheel: zoom image");
    println!("WASD: move around");
    println!("Space, Left-Shift: move up and down");
    println!("<- ->: switch models");
    println!("I: switch texture image");
    println!("R: toggle rotate");
    println!("T: toggle between random colors and texture");
    println!("L: reset camera and object");
    println!();

    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        toggle_rotate: true,
        ..Default::default()
    };
    app.model_carousel.set_dir("assets/models");
    app.image_carousel.set_dir("assets/images");
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
    load_prev_model: bool,
    load_next_model: bool,
    load_next_image: bool,
    is_left_clicked: bool,
    is_right_clicked: bool,
    cursor_position: Option<[i32; 2]>,
    cursor_delta: [i32; 2],
    wheel_delta: f32,
    tex_weight_change: f32,
    is_fullscreen: bool,

    model_carousel: Carousel,
    image_carousel: Carousel,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), anyhow::Error> {
        let window_attrs = Window::default_attributes()
            .with_title(TITLE)
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT));
        let window = event_loop.create_window(window_attrs).context("Failed to create window")?;

        let model_path = self.model_carousel.get_next(0, check_if_obj)
            .context("Failed to find a model")?;
        let nobj = NormalizedObj::from_reader(fs::load(model_path)?)?;

        let image_path = self.image_carousel.get_next(0, check_if_image)
            .context("Failed to find an image")?;
        let shader_spv = ShaderSpv {
            vert: include_bytes!(concat!(env!("OUT_DIR"), "/shader.vert.spv")),
            frag: include_bytes!(concat!(env!("OUT_DIR"), "/shader.frag.spv")),
        };
        let cubemap_spv = ShaderSpv {
            vert: include_bytes!(concat!(env!("OUT_DIR"), "/cubemap.vert.spv")),
            frag: include_bytes!(concat!(env!("OUT_DIR"), "/cubemap.frag.spv")),
        };
        let vulkan = VkApp::new(&window, WIDTH, HEIGHT, &image_path, nobj, shader_spv, cubemap_spv)?;

        self.vulkan = Some(vulkan);
        self.window = Some(window);
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(err) = self.init(event_loop) {
            log::error!("Error while starting: {err}");
            log::error!("{err:#?}");
            event_loop.exit();
        }
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
                        logical_key,
                        physical_key: PhysicalKey::Code(physical_key_code),
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();
                match physical_key_code {
                    KeyCode::KeyW => self.pressed.forward = pressed,
                    KeyCode::KeyA => self.pressed.left = pressed,
                    KeyCode::KeyS => self.pressed.backward = pressed,
                    KeyCode::KeyD => self.pressed.right = pressed,
                    KeyCode::Space => self.pressed.up = pressed,
                    KeyCode::ShiftLeft => self.pressed.down = pressed,
                    KeyCode::ArrowLeft if pressed => self.load_prev_model = true,
                    KeyCode::ArrowRight if pressed => self.load_next_model = true,
                    _ => {}
                }
                match logical_key {
                    Key::Character(key) if pressed && key == "f" => {
                        let fullscreen = if self.is_fullscreen {
                            None
                        } else {
                            Some(Fullscreen::Borderless(None))
                        };
                        self.window.as_mut().unwrap().set_fullscreen(fullscreen);
                        self.is_fullscreen = !self.is_fullscreen;
                    }
                    Key::Character(key) if pressed && key == "i"
                        => self.load_next_image = true,
                    Key::Character(key) if pressed && key == "r"
                        => self.toggle_rotate = !self.toggle_rotate,
                    Key::Character(key) if pressed && key == "l"
                        => self.vulkan.as_mut().unwrap().reset_ubo(),
                    Key::Character(key) if pressed && key == "t" => {
                        self.tex_weight_change = if self.tex_weight_change == 0. {
                            0.5 // change will take 2 secs from 0 to 1
                        } else {
                            -self.tex_weight_change
                        };
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized { .. } => {
                self.vulkan.as_mut().unwrap().dirty_swapchain = true;
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                self.is_left_clicked = state == ElementState::Pressed;
            }
            WindowEvent::MouseInput { button: MouseButton::Right, state, .. } => {
                self.is_right_clicked = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos: (i32, i32) = position.into();
                if self.is_left_clicked | self.is_right_clicked {
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
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if event_loop.exiting() {
            return;
        }

        if let Some((start, count)) = self.fps.as_mut() {
            let time = start.elapsed();
            *count += 1;
            if time.as_millis() > 1000 {
                use std::io::Write;

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
                app.recreate_swapchain(size.width, size.height);
            } else {
                return;
            }
        }

        let elapsed = self.last_frame.map(|instant| instant.elapsed()).unwrap_or_default();
        let delta = elapsed.as_secs_f32();
        self.last_frame = Some(Instant::now());

        let translation = Vector3::from([
            (self.pressed.left    as i8 - self.pressed.right    as i8) as f32 * delta,
            (self.pressed.down    as i8 - self.pressed.up       as i8) as f32 * delta,
            (self.pressed.forward as i8 - self.pressed.backward as i8) as f32 * delta,
        ]);
        app.view_matrix = Matrix4::from_translation(translation) * app.view_matrix;

        let extent = app.get_extent();
        let x_ratio = self.cursor_delta[0] as f32 / extent.width as f32;
        let y_ratio = self.cursor_delta[1] as f32 / extent.height as f32;
        if self.is_left_clicked {
            app.model_matrix = Matrix4::from_angle_y(Deg(x_ratio * 180.)) * app.model_matrix;
            app.model_matrix = Matrix4::from_angle_x(Deg(y_ratio * 180.)) * app.model_matrix;
        }
        if self.is_right_clicked {
            app.view_matrix = Matrix4::from_angle_y(Deg(x_ratio * 180.)) * app.view_matrix;
            app.view_matrix = Matrix4::from_angle_x(Deg(y_ratio * 180.)) * app.view_matrix;
        }
        if self.toggle_rotate {
            app.model_matrix = Matrix4::from_angle_y(Deg(delta * -90.)) * app.model_matrix;
        }
        self.cursor_delta = [0, 0];

        app.model_matrix = Matrix4::from_scale(1. + self.wheel_delta * 0.3) * app.model_matrix;
        self.wheel_delta = 0.;

        if self.load_next_model || self.load_prev_model {
            let offset = self.load_next_model as isize - self.load_prev_model as isize;
            match self.model_carousel.get_next(offset, check_if_obj) {
                Ok(path) => {
                    fn get_nobj(path: &Path) -> Result<NormalizedObj, anyhow::Error> {
                        Ok(NormalizedObj::from_reader(fs::load(path)?)?)
                    }
                    match get_nobj(&path) {
                        Ok(nobj) => app.load_new_model(nobj),
                        Err(err) => log::warn!("Failed to load model {}: {err}", path.display()),
                    }
                }
                Err(err) => log::warn!("Failed to find a model: {err}"),
            };
            self.load_next_model = false;
            self.load_prev_model = false;
        }
        if self.load_next_image {
            match self.image_carousel.get_next(1, check_if_image) {
                Ok(path) => {
                    if let Err(err) = app.load_new_texture(&path) {
                        log::warn!("Error while loading new image: {err}");
                        log::warn!("{err:#?}");
                    }
                }
                Err(err) => log::warn!("Failed to find an image: {err}"),
            };
            self.load_next_image = false;
        }

        app.texture_weight = (app.texture_weight + self.tex_weight_change * delta).clamp(0., 1.);

        app.dirty_swapchain = app.draw_frame();
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        if let Some(vulkan) = self.vulkan.as_ref() {
            vulkan.wait_gpu_idle();
        }
    }
}
