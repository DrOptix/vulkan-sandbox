use anyhow::Result;
use glfw::{Glfw, GlfwReceiver};

fn main() {
    match HelloTriangleApplication::new(800, 600, "Hello triangle") {
        Ok(mut app) => {
            app.run();
        }
        Err(err) => {
            eprint!("Error: {}", err);
        }
    };
}

struct HelloTriangleApplication {
    glfw: Glfw,
    window: glfw::PWindow,
    window_events: GlfwReceiver<(f64, glfw::WindowEvent)>,
}

/// Public functions
impl HelloTriangleApplication {
    pub fn new(width: u32, height: u32, title: &str) -> Result<Self> {
        let mut glfw = glfw::init_no_callbacks()?;
        let (window, window_events) = Self::create_window(&mut glfw, width, height, title)?;

        Ok(Self {
            glfw,
            window,
            window_events,
        })
    }

    pub fn run(&mut self) {
        self.window.set_key_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&self.window_events) {
                println!("{:?}", event);
                if let glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) = event {
                    self.window.set_should_close(true)
                }
            }
        }
    }
}

/// Internal functions
impl HelloTriangleApplication {
    fn create_window(
        glfw: &mut Glfw,
        width: u32,
        height: u32,
        title: &str,
    ) -> Result<(glfw::PWindow, GlfwReceiver<(f64, glfw::WindowEvent)>)> {
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(false));

        if let Some(window_and_events) =
            glfw.create_window(width, height, title, glfw::WindowMode::Windowed)
        {
            Ok(window_and_events)
        } else {
            anyhow::bail!("Unable to create GLFW window")
        }
    }
}
