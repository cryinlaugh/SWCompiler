add_library(swcc
			SHARED
			swcc_c_api.cpp)

target_compile_definitions(swcc INTERFACE -DCMAKE_CXX_FLAGS=-fPIC)
         
set_property(TARGET swcc PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(swcc 
	PRIVATE
	graphcore)
