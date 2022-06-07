if __name__ == "__main__":
    import vamtoolbox as vam
    
    vam.DLP.Setup.Focus(slices=20,N_screen=(1920,1080))

    vam.DLP.Setup.AxisAlignment(half_line_thickness=1,half_line_separation=200)
