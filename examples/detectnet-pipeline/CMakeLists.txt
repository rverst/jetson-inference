
file(GLOB detectnetCameraSources *.cpp)
file(GLOB detectnetCameraIncludes *.h )

cuda_add_executable(detectnet-pipeline ${detectnetCameraSources})

target_link_libraries(detectnet-pipeline jetson-inference)

install(TARGETS detectnet-pipeline DESTINATION bin)
