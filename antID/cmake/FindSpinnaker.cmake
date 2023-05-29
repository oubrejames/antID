# Locate Spinnaker library
find_path(SPINNAKER_INCLUDE_DIR NAMES Spinnaker.h PATHS /opt/spinnaker/include)
find_library(SPINNAKER_LIBRARY NAMES Spinnaker PATHS /opt/spinnaker/lib)

# Check if Spinnaker library and header files are found
if (SPINNAKER_INCLUDE_DIR AND SPINNAKER_LIBRARY)
    set(SPINNAKER_FOUND TRUE)
else()
    set(SPINNAKER_FOUND FALSE)
endif()

# Provide information about Spinnaker to CMake
if (SPINNAKER_FOUND)
    if (NOT Spinnaker_FOUND)
        set(Spinnaker_FOUND TRUE)
        set(Spinnaker_INCLUDE_DIRS ${SPINNAKER_INCLUDE_DIR})
        set(Spinnaker_LIBRARIES ${SPINNAKER_LIBRARY})
    endif()
endif()
message("SPINNAKER_FOUND: ${SPINNAKER_FOUND}")

# Provide variables to the user
mark_as_advanced(SPINNAKER_INCLUDE_DIR SPINNAKER_LIBRARY)

# # Locate Spinnaker library
# find_path(SPINNAKER_INCLUDE_DIR NAMES Spinnaker.h)
# find_library(SPINNAKER_LIBRARY NAMES Spinnaker)

# message("SPINNAKER_INCLUDE_DIR: ${SPINNAKER_INCLUDE_DIR}")
# message("SPINNAKER_LIBRARY: ${SPINNAKER_LIBRARY}")

# set(SPINNAKER_INCLUDE_DIR ${SPINNAKER_INCLUDE_DIR})
# set(SPINNAKER_LIBRARY ${SPINNAKER_LIBRARY})

# include(find_package(PackageHandleStandardArgs))

# find_package_handle_standard_args(SPINNAKER DEFAULT_MSG SPINNAKER_INCLUDE_DIR SPINNAKER_LIBRARY)

# # Provide variables to the user
# mark_as_advanced(SPINNAKER_INCLUDE_DIR SPINNAKER_LIBRARY)