use std::{
    borrow::Cow,
    ffi::{CStr, CString, c_char},
};

use anyhow::{Context, Result};
use ash::{ext::debug_utils, vk};
use glfw::{Glfw, GlfwReceiver};

fn main() {
    match HelloTriangleApplication::new(800, 600, "Hello triangle") {
        Ok(mut app) => {
            app.run();
        }
        Err(err) => {
            eprint!("Error: {err}");
        }
    };
}

struct HelloTriangleApplication {
    glfw: Glfw,
    window: glfw::PWindow,
    window_events: GlfwReceiver<(f64, glfw::WindowEvent)>,
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_instance: Option<debug_utils::Instance>,
    debug_utils_messanger: Option<vk::DebugUtilsMessengerEXT>,
}

/// Public functions
impl HelloTriangleApplication {
    pub fn new(width: u32, height: u32, title: &str) -> Result<Self> {
        let mut glfw = glfw::init_no_callbacks()?;
        let (window, window_events) = Self::create_window(&mut glfw, width, height, title)?;

        let mut extensions = glfw
            .get_required_instance_extensions()
            .context("Vulkan is not supported")?;

        let mut validation_layers = vec![];
        if cfg!(debug_assertions) {
            extensions.push("VK_EXT_debug_utils".to_string());
            validation_layers.push("VK_LAYER_KHRONOS_validation".to_string());
        }

        let entry = unsafe { ash::Entry::load() }?;
        let instance = Self::create_instance(&entry, &extensions, &validation_layers)?;
        let maybe_debug_messanger = Self::setup_debug_messenger(&entry, &instance)?;
        let debug_utils_instance = maybe_debug_messanger.as_ref().map(|debug| debug.0.clone());
        let debug_utils_messanger = maybe_debug_messanger.as_ref().map(|debug| debug.1);

        Ok(Self {
            glfw,
            window,
            window_events,
            _entry: entry,
            instance,
            debug_utils_instance,
            debug_utils_messanger,
        })
    }

    pub fn run(&mut self) {
        self.window.set_key_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&self.window_events) {
                println!("{event:?}");
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
        extensions: &[String],
        validation_layers: &[String],
    ) -> Result<ash::Instance> {
        if cfg!(debug_assertions) {
            Self::check_validation_layers_support(entry, validation_layers)?;
        }

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3)
            .engine_name(c"hello_trangle")
            .engine_version(0)
            .application_name(c"hello_trangle")
            .engine_version(0);

        let extensions: Vec<CString> = extensions
            .iter()
            .map(|ext| CString::new(ext.as_str()).unwrap())
            .collect();

        let extensions: Vec<*const c_char> = extensions
            .iter()
            .map(|ext| ext.as_ptr() as *const c_char)
            .collect();

        let validation_layers: Vec<CString> = validation_layers
            .iter()
            .map(|layer| CString::new(layer.as_str()).unwrap())
            .collect();

        let validation_layers: Vec<*const c_char> = validation_layers
            .iter()
            .map(|layer| layer.as_ptr() as *const c_char)
            .collect();

        let create_info = vk::InstanceCreateInfo::default()
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&validation_layers)
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(instance)
    }

    fn check_validation_layers_support(
        entry: &ash::Entry,
        requested_layers: &[String],
    ) -> Result<()> {
        let existing_layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .context("Failed to enumerate instance layer properties")?
        };

        for layer in requested_layers {
            let mut found = false;
            for prop in &existing_layer_properties {
                if unsafe {
                    CStr::from_ptr(prop.layer_name.as_ptr())
                        .to_str()
                        .context("Failed to convert layer name to string")
                }?
                .eq(layer.as_str())
                {
                    found = true;
                    break;
                }
            }

            if !found {
                anyhow::bail!("Validation layer '{}' not found", layer);
            }
        }

        Ok(())
    }

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>> {
        if cfg!(debug_assertions) {
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(Self::vulkan_debug_callback));

            let debug_utils_instance = debug_utils::Instance::new(entry, instance);
            let debug_utils_messanger =
                unsafe { debug_utils_instance.create_debug_utils_messenger(&create_info, None)? };

            return Ok(Some((debug_utils_instance, debug_utils_messanger)));
        }

        Ok(None)
    }

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        unsafe {
            let callback_data = *p_callback_data;
            let message_id_number = callback_data.message_id_number;

            let message_id_name = if callback_data.p_message_id_name.is_null() {
                Cow::from("")
            } else {
                CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
            };

            let message = if callback_data.p_message.is_null() {
                Cow::from("")
            } else {
                CStr::from_ptr(callback_data.p_message).to_string_lossy()
            };

            println!(
                "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
            );

            vk::FALSE
        }
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_utils_instance) = self.debug_utils_instance.as_ref() {
                if let Some(debug_utils_messanger) = self.debug_utils_messanger {
                    debug_utils_instance.destroy_debug_utils_messenger(debug_utils_messanger, None);
                }
            }
            self.instance.destroy_instance(None);
        };
    }
}
