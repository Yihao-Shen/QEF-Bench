# Add this fragment from OCUDU's LDPC benchmark CMakeLists.txt after copying
# the two corresponding .cpp sources into that directory.
foreach(name ldpc_decoder_energy_benchmark_cb_cells ldpc_decoder_single_cb_scan)
  if(NOT TARGET ${name})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ocudu_channel_coding ocudulog ocudu_support)
  endif()
endforeach()
