def enumerate_lat_unc(COORD_PREC):
    if COORD_PREC == 0:
        return 5e-6
    elif COORD_PREC == 1:
        return 0.01
    elif COORD_PREC == 10:
        return 0.1
    elif COORD_PREC == 60:
        return 0.5
    elif COORD_PREC == 7 or COORD_PREC == 11:
        return 0.25
    else:
        raise ValueError(f"Unrecognized COORD_PREC value: {COORD_PREC}")