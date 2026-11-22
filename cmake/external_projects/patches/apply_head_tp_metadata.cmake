# FetchContent may repeat the patch step in a previously configured checkout.
execute_process(COMMAND git apply --reverse --check "${PATCH_FILE}"
                RESULT_VARIABLE already_applied ERROR_QUIET)
if(NOT already_applied EQUAL 0)
  execute_process(COMMAND git apply "${PATCH_FILE}" COMMAND_ERROR_IS_FATAL ANY)
endif()
