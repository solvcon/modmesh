find_program(flake8_BIN NAMES flake8)
if(NOT flake8_BIN)
    message(ERROR "flake8 not found, run pip install flake8")
else()
    message(STATUS "using program '${flake8_BIN}'")
endif()

macro(flake8 target_name)
    add_custom_command(TARGET ${target_name}
        PRE_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${flake8_BIN} . --exclude tmp
        COMMENT "Running flake8 on ${CMAKE_CURRENT_SOURCE_DIR} ...")
endmacro()