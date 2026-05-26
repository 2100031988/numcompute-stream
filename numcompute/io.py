import numpy as np

def load_csv(         # converts csv file to numpy arrays while handling missing values, hanbdling different data types, errors and formating 
                        # like skipping the rows and replacing the values with 0 as initial value.

    filepath,          
    delimiter=",",
    dtype=float,
    missing_values=None,
    fill_value=np.nan,
    skip_header=0
):
    
    if missing_values is None:              # it is same as numpy formating where we handle missign values in rows and columns (to be specific) 
                                            # and here we have default values for missing values as empty string, NA, null and None and we can specify our own data strings as well.

        missing_values = ["", "NA", "null", "None"]


    try:                                    # reades csv file and detects abd entries, replaces them and then fill then skips column names and also remove spaces using the parameter "autostrip=True".
        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=fill_value,
            skip_header=skip_header,
            autostrip=True
        )

    except OSError:                                                 # if file is not found or there is an error in reading the file then we raise a "FileNotFoundError" with a message indicating the file path that was not found.

        raise FileNotFoundError(f"File not found: {filepath}")


    if data is None or data.size == 0:                      # if the data is empty or invalid then we raise a "ValueError" with a message indicating that the CSV file is empty or invalid.

        raise ValueError("The given CSV file is empty or invalid. Please provide a valid CSV file with specified data.")


    if data.ndim == 1:                                  # this is required to fix the shape of the data and uses "reshape" function where we convert a 1D array to a 2D array with one column.
        data = data.reshape(-1, 1)


    if np.isnan(data).all():                    # if all values in the data are missing (NaN) then we raise a "ValueError" with a message indicating that all values are missing and ask the user to provide a CSV file with some valid data.
        raise ValueError("All values are missing. Please provide a CSV file with some valid data.")

    return data
