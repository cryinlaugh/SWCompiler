file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} utils/*)
file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(TESTSRC ${TEST_SOURCES})
   get_filename_component(TESTNAME ${TESTSRC} NAME_WE) # file name without directories or longest extension
   add_executable(${TESTNAME} ${TESTSRC})
   install(TARGETS ${TESTNAME} RUNTIME DESTINATION bin)
endforeach()

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} input/*)
foreach(filename ${files})
  configure_file(${filename} ${SW_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)
