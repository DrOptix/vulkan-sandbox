use std::{
    borrow::Cow,
    ffi::{CStr, CString, c_char},
    ptr,
};

use anyhow::{Context, Result};
use ash::{
    ext::debug_utils,
    vk::{self, PhysicalDevice},
};
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

#[derive(Debug, Default)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

struct HelloTriangleApplication {
    glfw: Glfw,
    window: glfw::PWindow,
    window_events: GlfwReceiver<(f64, glfw::WindowEvent)>,
    _window_surface: vk::SurfaceKHR,
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_instance: Option<debug_utils::Instance>,
    debug_utils_messanger: Option<vk::DebugUtilsMessengerEXT>,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    _graphics_queue: vk::Queue,
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
        let physical_device = Self::pick_physical_device(&instance)?;
        let (device, graphics_queue) = Self::create_logical_device(&instance, &physical_device)?;
        let window_surface = Self::create_window_surface(&instance, &window)?;

        Ok(Self {
            glfw,
            window,
            window_events,
            _window_surface: window_surface,
            _entry: entry,
            instance,
            debug_utils_instance,
            debug_utils_messanger,
            _physical_device: physical_device,
            device,
            _graphics_queue: graphics_queue,
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

    fn create_window_surface(
        instance: &ash::Instance,
        window: &glfw::PWindow,
    ) -> Result<vk::SurfaceKHR> {
        let mut surface: std::mem::MaybeUninit<vk::SurfaceKHR> = std::mem::MaybeUninit::uninit();

        if window.create_window_surface(instance.handle(), ptr::null(), surface.as_mut_ptr())
            != vk::Result::SUCCESS
        {
            anyhow::bail!("Failed to create GLFW Vulkan window surface.");
        }

        Ok(unsafe { surface.assume_init() })
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
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
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

    fn pick_physical_device(instance: &ash::Instance) -> Result<PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        if physical_devices.is_empty() {
            anyhow::bail!("Failed to find devices with Vulkan support");
        }

        let device = physical_devices
            .into_iter()
            .find(|dev| Self::is_device_suitable(instance, dev));

        if let Some(device) = device {
            Ok(device)
        } else {
            anyhow::bail!("Unable to find a suitable device");
        }
    }

    fn is_device_suitable(instance: &ash::Instance, physical_device: &vk::PhysicalDevice) -> bool {
        let queue_family_indices = Self::find_queue_families(instance, physical_device);
        queue_family_indices.is_complete()
    }

    fn find_queue_families(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> QueueFamilyIndices {
        let mut queue_family_indices = QueueFamilyIndices::default();

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

        for (index, properties) in queue_family_properties.iter().enumerate() {
            if properties.queue_flags.intersects(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        queue_family_indices
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> Result<(ash::Device, vk::Queue)> {
        let queue_family_indices = Self::find_queue_families(instance, physical_device);

        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_priorities(std::slice::from_ref(&1.0))
            .queue_family_index(queue_family_indices.graphics_family.unwrap());

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .enabled_features(&device_features)
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device = unsafe {
            instance
                .create_device(*physical_device, &device_create_info, None)
                .context("Failed to create logical device")?
        };

        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) };

        Ok((device, graphics_queue))
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
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        };
    }
}
