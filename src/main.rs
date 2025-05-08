use std::ffi::c_char;

use anyhow::{Context, Result};
use ash::vk;
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
    _entry: ash::Entry,
    instance: ash::Instance,
}

/// Public functions
impl HelloTriangleApplication {
    pub fn new(width: u32, height: u32, title: &str) -> Result<Self> {
        let mut glfw = glfw::init_no_callbacks()?;
        let (window, window_events) = Self::create_window(&mut glfw, width, height, title)?;
        let glfw_required_extensions = glfw
            .get_required_instance_extensions()
            .context("Vulkan is not supported")?;

        let entry = unsafe { ash::Entry::load() }?;
        let instance = Self::create_instance(&entry, &glfw_required_extensions)?;

        Ok(Self {
            glfw,
            window,
            window_events,
            _entry: entry,
            instance,
        })
    }

    pub fn run(&mut self) {
        self.window.set_key_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&self.window_events) {
                println!("{:?}", event);
                if let glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) = event
                {
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

    fn create_instance(
        entry: &ash::Entry,
        required_extensions: &[String],
    ) -> Result<ash::Instance> {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3)
            .engine_name(c"hello_trangle")
            .engine_version(0)
            .application_name(c"hello_trangle")
            .engine_version(0);

        let extension_pointers: Vec<*const c_char> = required_extensions
            .iter()
            .map(|ext| ext.as_ptr() as *const c_char)
            .collect();

        let create_info = vk::InstanceCreateInfo::default()
            .enabled_extension_names(&extension_pointers)
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(instance)
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}
