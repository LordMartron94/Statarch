package core

import (
	"echo"
	"essence"
)

const uuidStr = "ddf805d0-6f90-4c0c-a1f4-7efd2c95a3ff"

var StatarchUUID essence.UUID

func init() {
	id, _ := essence.UUIDFromString(uuidStr)
	StatarchUUID = id

	echo.EchoSystemRegister(
		StatarchUUID,
		echo.EchoSystemConfiguration{
			MinLogLevel:    echo.TRACE,
			SystemPrefixes: []string{"Statarch"},
		},
	)
}
