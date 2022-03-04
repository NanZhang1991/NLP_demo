     try:
          error
     except Exception as e:
        error_line = e.__traceback__.tb_lineno
        traceback.format_exc()
