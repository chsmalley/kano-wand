'''
Known Peripheral UUIDs, obtained by querying using the Bluepy module:
=====================================================================
Anti DOS Characteristic <00020005-574f-4f20-5370-6865726f2121>
Battery Level Characteristic <Battery Level>
Peripheral Preferred Connection Parameters Characteristic <Peripheral Preferred Connection Parameters>
API V2 Characteristic <00010002-574f-4f20-5370-6865726f2121>
DFU Control Characteristic <00020002-574f-4f20-5370-6865726f2121>
Name Characteristic <Device Name>
Appearance Characteristic <Appearance>
DFU Info Characteristic <00020004-574f-4f20-5370-6865726f2121>
Service Changed Characteristic <Service Changed>
Unknown1 Characteristic <00020003-574f-4f20-5370-6865726f2121>
Unknown2 Characteristic <00010003-574f-4f20-5370-6865726f2121>

The rest of the values saved in the dictionaries below, were borrowed from
@igbopie's javacript library, which is available at https://github.com/igbopie/spherov2.js

'''

from typing import NamedTuple

API_V2_CHARACTERISTIC = "00010002-574f-4f20-5370-6865726f2121"
ANTI_DOS_CHARACTERISTIC = "00020005-574f-4f20-5370-6865726f2121"
DFU_CHARACTERISTIC = "00020002-574f-4f20-5370-6865726f2121"
DFU2_CHARACTERISTIC = "00020004-574f-4f20-5370-6865726f2121"
UNKNOWN1_CHARACTERISTIC = "00020003-574f-4f20-5370-6865726f2121"
UNKNOWN2_CHARACTERISTIC = "00010003-574f-4f20-5370-6865726f2121"

class DEVICE_ID(NamedTuple):
    api_processor: 0x10
    system_info: 0x11
    power_info: 0x13
    driving: 0x16
    animatronics: 0x17
    sensor: 0x18
    something: 0x19
    user_io: 0x1a
    something_api: 0x1f

class SYSTEM_INFO_COMMANDS(NamedTuple):
    mainApplicationVersion: 0x00
    bootloaderVersion: 0x01
    something: 0x06
    something2: 0x13
    something6: 0x12
    something7: 0x28

class SEND_PACKET_CONSTANTS(NamedTuple):
    StartOfPacket: 0x8d
    EndOfPacket: 0xd8

class USER_IO_COMMAND_IDS(NamedTuple):
    all_leds: 0x0e

class FLAGS(NamedTuple):
    isResponse: 0x01
    requestsResponse: 0x02
    requestsOnlyErrorResponse: 0x04
    resetsInactivityTimeout: 0x08

class POWER_COMMAND_IDS(NamedTuple):
    deepSleep: 0x00
    sleep: 0x01
    batteryVoltage: 0x03
    wake: 0x0D
    something: 0x05
    something2: 0x10
    something3: 0x04
    something4: 0x1E

class DRIVING_COMMANDS(NamedTuple):
    rawMotor: 0x01
    resetHeading: 0x06
    driveAsSphero: 0x04
    driveAsRc: 0x02
    driveWithHeading: 0x07
    stabilization: 0x0C

class SENSOR_COMMANDS(NamedTuple):
    sensorMask: 0x00
    sensorResponse: 0x02
    configureCollision: 0x11
    collisionDetectedAsync: 0x12
    resetLocator: 0x13
    enableCollisionAsync: 0x14
    sensor1: 0x0F
    sensor2: 0x17
    configureSensorStream: 0x0C
