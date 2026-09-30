# Setup process and ROS code for using Unitree GO2 EDU quadruped with essential ROS functionality. Functionalities include pointcloud and laserscan acquisition, SLAM, Nav2 navigation, camera data, etc.
### This guide assumes you have a GO2 or other robot equipped with an Intel Realsense D435i and a Livox LiDar
### and that both of these components have been mounted on the robot following the official docs.
### It also assumes the robot is running ROS2 Foxy, but this shouldn't cause many issues.

## Go2 Expansion Board Networking Setup

This section outlines how to get the expansion board component of the Go2 dog (an NVIDIA Orin board) connected to an external Wi-Fi network and how to access the ROS topics of the Go2's internal board on the expansion board. This is a separate process than connecting the Go2 to Wi-Fi as outlined in the official Unitree docs. Before starting this section, you will need:
1. A monitor, keyboard, and mouse
2. A BrosTrend AC1L USB Wi-Fi adapter (or equivalent Ubuntu-compatible Wi-Fi adapter)
3. A Type-C USB hub with an Ethernet port 
4. An HDMI cable that can be plugged into the aforementioned USB hub
5. An Ubuntu host PC with Ethernet port or equivalent

### Step 1: 
Follow the quick start section of the Unitree docs as found in the [Unitree Docs](https://support.unitree.com/home/en/developer/Quick_start) and the app binding section. **Ensure you connect to the dog through the app through the Wi-Fi mode**. Once connected through the app, record the Go2's IP (Device/Data/Automatic Machine Inspection) as it will be needed later.

It's also a good idea to go ahead and do the [Go2 ROS2 setup](https://support.unitree.com/home/en/developer/ROS2_service) that matches with your distro.

### Step 2:
Plug the USB hub into the Type-C port on the back of the Go2 expansion board. Plug the HDMI into the USB hub, and into the monitor. Connect the keyboard and mouse to the USB hub. The monitor should display the Ubuntu GUI. Login to the expansion board (default password is 123).

Go to the network settings. **Disable the PCI Ethernet connection**. This will allow the board to connect to the Internet. Plug the Ethernet cable into the USB hub, and then into an Internet-connected USB port. Check if the expansion board can connect to the Internet, e.g. by `ping google.com`.

### Step 3:
Download the BrosTrend AC1L adapter Linux drivers (or the equivalent for your Wi-Fi adapter) by running

```
sudo sh -c 'wget linux.brostrend.com/install -O /tmp/install && sh /tmp/install'
```

Follow the instructions in the terminal. Ensure all drivers are installed correctly.

### Step 4:
Once the drivers have been installed, go back to network settings, and connect to your Wi-Fi network of choice. Unplug the Ethernet cable from the USB hub, and ensure you can access the Internet on the Wi-Fi connection. Record the IP of the expansion board on the Wi-Fi so you can acces it later. If there are no problems, **re-enable the PCI Ethernet connection**; this will resume the communication between the expansion board and the Go2's internal board. 

In network settings, open the wired connection settings and change the DHCP settings IPv4 address to `192.168.123.99` and make sure the mask is `255.255.255.0`. This will allow you to access the topics published by the Go2's internal board on the expansion board.


### Step 5:
Run the commands `echo 'net.core.rmem_max=52428800' | sudo tee -a /etc/sysctl.conf` and `sudo sysctl -p`. This will increase the expansion board's maximum socket buffer size. To utilize this with ROS, go to `~/cyclonedds_ws/cyclonedds.xml` and add the following lines if they are not there already:
```
<Internal>
    <MinimumSocketReceiveBufferSize>50MB</MinimumSocketReceiveBufferSize>
</Internal>
```
This will allow the expansion board to handle more data across its Ethernet bus.

Finally, increase the power level of the expansion board to 25W or higher. 

## D1 Arm Setup

This section covers setup for the D1 arm for the Go2. 

### Physical Setup

1. Power off the Go2 before connecting or disconnecting the arm.
2. Mount the D1 securely using the Unitree payload mounting instructions.
3. Connect the arm's power lead to the dedicated external robotic-arm power
   connector near the Go2 battery/expansion-dock power connection. **Do not
   connect the arm to the expansion dock's 12 V or 5 V accessory outputs.**
4. Connect the arm's communication lead to the expansion dock's
   GH1.25-4-pin Ethernet port.
5. Secure both cables so they cannot contact the arm joints or become pinched
   while the Go2 or arm moves, then power the Go2 back on.

For the mounting location and connector layout, refer to Unitree's
[Payload instructions](https://support.unitree.com/home/en/developer/Payload).




## Go2 Expansion Board Intel RealSense Setup
The Go2's expansion board does not come preconfigured to use the D435i depth camera with Python bindings. Since the board uses an ARM architecture, the bindings can't be installed directly through pip either, and must be built from source. Steps 1-4 outlined below are taken directly from [this official process from a RealSense Git issue](https://github.com/IntelRealSense/librealsense/issues/6964). Steps 5-6 were added by me. The process should be nearly identical for any ARM system, and should be even simpler if your target platform does not have an ARM architecture.

### Step 1:
The first step in this process is to download the source code of librealsense from the Releases page as a zip file. The source code zip file can always be found in the "Assets" list at the bottom of the information listing for each SDK version on the Releases page.

https://github.com/IntelRealSense/librealsense/releases/

### Step 2: 
After downloading the zip file, its contents should be extracted so that you have a librealsense folder.At this point you can use the RSUSB installation method to install librealsense without dependence on Linux versions or kernel versions and without the need for patching. This makes this installation method particularly suited to Arm devices such as Jetson.

The backend installation method requires an internet connection. The steps are:

Go to the librealsense root directory. Create a folder within it called build and then go to this new folder using this instruction: `mkdir build && cd build`

### Step 3:
Now that you are in the build directory, run the CMake build instruction below to install librealsense and the Python bindings over the internet connection:
`cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=...`

The above statement is a basic one that should test whether the build is likely to succeed or not. **(Replace the dots with the path to your Python exec)**

### Step 4:
If it does succeed then you can try a more advanced build, which builds the example programs and includes optimizations such as building with CUDA support for faster alignment processing on devices such as Jetson that are equipped with an Nvidia graphics GPU.

`cmake ../-DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=... -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false -DBUILD_WITH_CUDA:bool=true`

(The original issue has this command running with examples and graphical examples on. This can cause issues with the following step, so they have been disabled here.)

### Step 5:
Run `make -j$(nproc)`

### Step 6: 
If everything builds correctly, run `find . -name "pyrealsense2*.so"` within the build directory to file the .so filepath. Once you have this path, find the site packages location with `python3 -m site --user-site`. Then, use `cp` to copy the .so to the site packages location.

## Livox Setup
Most of this process is taken care of by mounting the Livox according to the official docs from Unitree or whatever the target platform is. The interfacing of the Livox with the robot should be taken care of by the packages in this repo. However, you may need to change the `ip` parameter within the `src/CPSL_ROS_livox_ros_driver2/config/HAP_config.json` file to the IP of your Livox device (IP can be found on a sticker on the physical device).

## Go2 Internal IP Configuration
For the Go2 to receive velocity commands from packages such as Nav2, the `cmd_vel` messages must be translated and forwarded to the Go2’s internal board, which handles the robot’s movement. The `CmdVelTranslator.py` node in the `dog_utilities` package therefore accepts an `internal_board_ip` parameter, which can be specified either when launching the main launch file or when running the node directly.

By default, this parameter is set to the internal board’s Ethernet IP address because the internal and external boards are connected by Ethernet. Alternatively, if the internal board is connected to the same Wi-Fi network as the external board, its Wi-Fi IP address can be supplied instead. This is useful if for some reason you're sending commands and running the translator script from a device that is not the external board.


## Using this repo
1. Clone this repo and build and source the workspace
2. Open 3 terminal windows and run the following
    * `ros2 launch launcher dog.launch.py internal_board_ip:=OPTIONAL_YOUR_IP_HERE collect_realsense:=true or false launch_arm:= true or false arm_command_enabled:=true or false`
    * `ros2 launch cpsl_ros2_sensors_bringup ugv_sensor_bringup.launch.py`
    * `ros2 launch cpsl_nav slam.launch.py scan_topic:=/livox/scan_best_effort`
3. For getting a functional transform tree and mapping, that's all you need. If you want to issue Nav2 commands, open another terminal and run
    * `ros2 launch cpsl_nav nav2_archived.py scan_topic:=/livox/scan_best_effort`

## Unitree D1 arm

The `unitree_arm` package defines the D1 `ArmString` wire message and typed
feedback/service interfaces. The `unitree_arm_control` package adds a
passive-by-default ROS 2 wrapper for parsing `/arm_Feedback` and safely
publishing validated commands to `/arm_Command`. It communicates through the
existing ROS 2/CycloneDDS topics and does not require changes to Unitree SDK2.

See [`src/unitree_arm_control/README.md`](src/unitree_arm_control/README.md) for
build, feedback verification, launch, and command examples.

