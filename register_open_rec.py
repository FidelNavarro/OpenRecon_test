import os
import logging
import traceback
import base64
from time import perf_counter

import numpy as np
import numpy.fft as fft
import ismrmrd
import ctypes

# helper modules (provided by the environment)
import mrdhelper
import constants

import SimpleITK as sitk  # for registration

# Folder for debug output files
debugFolder = "/tmp/share/debug"

# Registration helpers
def register_images(fixed_np: np.ndarray,
                    moving_np: np.ndarray,
                    voxel_spacing: tuple = (1., 1., 1.),
                    return_transform: bool = False):
    """
    Register 'moving_np' to 'fixed_np' (both [z, y, x]) using SimpleITK MI + RSGD.
    Returns the registered moving image as numpy array [z, y, x].
    """
    fixed  = sitk.GetImageFromArray(fixed_np)
    moving = sitk.GetImageFromArray(moving_np)

    fixed  = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    fixed.SetSpacing(voxel_spacing)
    moving.SetSpacing(voxel_spacing)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving, fixed, final_transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    registered_np = sitk.GetArrayFromImage(resampled)

    if return_transform:
        return registered_np, final_transform
    return registered_np


def _pick_ref_by_bvalue(meta_list):
    """
    Return (index, bval) of the lowest b-value from ISMRMRD Meta
    (keys 'b-value' or 'BValue'). Fallback to (0, None).
    """
    bvals = []
    for m in meta_list:
        b = m.get('b-value')
        if b is None:
            b = m.get('BValue')
        try:
            bvals.append(float(b) if b is not None else np.nan)
        except Exception:
            bvals.append(np.nan)
    if np.all(np.isnan(bvals)):
        return 0, None
    iref = int(np.nanargmin(bvals))
    return iref, bvals[iref]


def _spacing_from_head(h):
    """Compute voxel spacing (dx, dy, dz) [mm] from ISMRMRD image header (FOV/matrix)."""
    mx  = (int(h.matrix_size[0]), int(h.matrix_size[1]), max(int(h.matrix_size[2]), 1))
    fov = (float(h.field_of_view[0]), float(h.field_of_view[1]), float(h.field_of_view[2]))
    dx = fov[0] / mx[0] if mx[0] > 0 else 1.0
    dy = fov[1] / mx[1] if mx[1] > 0 else 1.0
    dz = fov[2] / mx[2] if mx[2] > 0 else 1.0
    return (dx, dy, dz)

def process(connection, config, mrdHeader):
    logging.info("Config: \n%s", config)

    try:
        logging.info("Incoming dataset contains %d encodings", len(mrdHeader.encoding))
        logging.info(
            "First encoding: traj='%s', matrix=(%s x %s x %s), FOV=(%s x %s x %s) mm",
            mrdHeader.encoding[0].trajectory,
            mrdHeader.encoding[0].encodedSpace.matrixSize.x,
            mrdHeader.encoding[0].encodedSpace.matrixSize.y,
            mrdHeader.encoding[0].encodedSpace.matrixSize.z,
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.x,
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.y,
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z,
        )
    except Exception:
        logging.info("Improperly formatted MRD header: \n%s", mrdHeader)

    currentSeries = 0
    acqGroup = []
    imgGroup = []
    waveformGroup = []

    try:
        for item in connection:
            if isinstance(item, ismrmrd.Acquisition):
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
                    acqGroup.append(item)

                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, connection, config, mrdHeader)
                    connection.send_image(image)
                    acqGroup = []

            elif isinstance(item, ismrmrd.Image):
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images (series index changed to %d)", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, mrdHeader)
                    connection.send_image(image)
                    imgGroup = []

                # Only process magnitude / unknown type (0)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry'] = 1
                    item.attribute_string = tmpMeta.serialize()
                    connection.send_image(item)

            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Process any remaining data
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, connection, config, mrdHeader)
            connection.send_image(image)
            acqGroup = []

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, mrdHeader)
            connection.send_image(image)
            imgGroup = []

    except Exception:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()

def process_raw(acqGroup, connection, config, mrdHeader):
    if len(acqGroup) == 0:
        return []

    logging.info('-----------------------------------------------')
    logging.info('     process_raw called with %d readouts', len(acqGroup))
    logging.info('-----------------------------------------------')

    tic = perf_counter()

    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)

    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in acqGroup]
    phs = [acquisition.idx.phase                for acquisition in acqGroup]

    data = np.zeros((acqGroup[0].data.shape[0],
                     mrdHeader.encoding[0].encodedSpace.matrixSize.y,
                     mrdHeader.encoding[0].encodedSpace.matrixSize.x,
                     max(phs) + 1),
                    acqGroup[0].data.dtype)

    rawHead = [None] * (max(phs) + 1)

    for acq, lin, phs in zip(acqGroup, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            data[:, lin, -acq.data.shape[1]:, phs] = acq.data

            # center line in user[5]
            if (rawHead[phs] is None) or (
                abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) <
                abs(rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])
            ):
                rawHead[phs] = acq.getHead()

    # Flip to be consistent with ICE
    data = np.flip(data, (1, 2))

    # FFT
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    data *= np.prod(data.shape)

    # SOS coil combine -> [PE RO phs]
    data = np.sqrt(np.sum(np.square(np.abs(data)), axis=0))

    # Remove oversampling in RO
    if mrdHeader.encoding[0].reconSpace.matrixSize.x != 0:
        offset = int((data.shape[1] - mrdHeader.encoding[0].reconSpace.matrixSize.x) / 2)
        data = data[:, offset:offset + mrdHeader.encoding[0].reconSpace.matrixSize.x]

    # Remove oversampling in PE
    if mrdHeader.encoding[0].reconSpace.matrixSize.y != 0:
        offset = int((data.shape[0] - mrdHeader.encoding[0].reconSpace.matrixSize.y) / 2)
        data = data[offset:offset + mrdHeader.encoding[0].reconSpace.matrixSize.y, :]

    # Time
    toc = perf_counter()
    connection.send_logging(constants.MRD_LOGGING_INFO, "Total processing time: %.2f ms" % ((toc - tic) * 1000.0))

    # Format as ISMRMRD images
    imagesOut = []
    for ph in range(data.shape[2]):
        # Use transpose=False to avoid PendingDeprecationWarning
        tmpImg = ismrmrd.Image.from_array(data[..., ph], transpose=False)

        # Header from raw
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[ph]))
        tmpImg.field_of_view = (ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.x),
                                ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = ph

        # Meta
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['Keep_image_geometry']    = 1
        tmpImg.attribute_string = tmpMeta.serialize()

        imagesOut.append(tmpImg)

    # Register & return
    imagesOut = process_image(imagesOut, connection, config, mrdHeader)
    return imagesOut

def process_image(imgGroup, connection, config, mrdHeader):
    if len(imgGroup) == 0:
        return []

    logging.info('-----------------------------------------------')
    logging.info('     process_image called with %d images', len(imgGroup))
    logging.info('-----------------------------------------------')

    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)

    logging.debug("Input dtype: %s", ismrmrd.get_dtype_from_data_type(imgGroup[0].data_type))

    # MRD stores data as [cha, z, y, x]
    # Stack → [img, cha, z, y, x], Heads/Meta lists
    data = np.stack([img.data for img in imgGroup])                 # [img, cha, z, y, x]
    head = [img.getHead() for img in imgGroup]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in imgGroup]

    # Reformat to [y, x, z, cha, img]
    data = data.transpose((3, 4, 2, 1, 0))

    # ---------------------- Registration to b0 ----------------------
    iref, ref_b = _pick_ref_by_bvalue(meta)
    logging.info("Registration reference image index: %d (b=%s)", iref, str(ref_b))

    # Voxel spacing from header
    dx, dy, dz = _spacing_from_head(head[iref])

    # Register per image (per channel if needed)
    zdim = data.shape[2]
    ch   = data.shape[3]
    nimg = data.shape[4]

    # Fixed reference (channel 0) as [z, y, x]
    fixed_vol = data[:, :, :, 0, iref].transpose(2, 0, 1).astype(np.float32)

    data_reg = np.copy(data)
    for ii in range(nimg):
        if ii == iref:
            continue
        for c in range(ch):
            moving_vol = data[:, :, :, c, ii].transpose(2, 0, 1).astype(np.float32)
            reg_vol = register_images(fixed_vol, moving_vol, voxel_spacing=(dx, dy, dz))
            data_reg[:, :, :, c, ii] = reg_vol.transpose(1, 2, 0)

    data = data_reg

    # ---------------------- Intensity handling ----------------------
    # Default: output int16 magnitude (no contrast inversion)
    if mrdhelper.get_json_config_param(config, 'options') == 'complex':
        data = data.astype(np.complex64)
        maxVal = np.max(np.abs(data)) if data.size > 0 else 0
    else:
        BitsStored = 12
        if (mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored") is not None):
            BitsStored = mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored")
        maxVal = 2**BitsStored - 1

        data = np.abs(data).astype(np.float64)
        peak = np.max(data) if data.size > 0 else 0
        if peak > 0:
            data *= maxVal / peak
        data = np.around(data).astype(np.int16)

    # ---------------------- Back to MRD Images ----------------------
    imagesOut = [None] * nimg
    for iImg in range(nimg):
        # data[..., iImg] is [y, x, z, cha] → [cha, z, y, x]
        cz_yx = data[..., iImg].transpose((3, 2, 0, 1))
        imagesOut[iImg] = ismrmrd.Image.from_array(cz_yx, transpose=False)

        # Copy/adjust header
        oldHeader = head[iImg]
        oldHeader.data_type = imagesOut[iImg].data_type

        if (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXFLOAT) or (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXDOUBLE):
            oldHeader.image_type = ismrmrd.IMTYPE_COMPLEX

        imagesOut[iImg].setHead(oldHeader)

        # Meta: tag as registered, keep geometry, set windowing for viewers
        tmpMeta = meta[iImg]
        iph = tmpMeta.get('ImageProcessingHistory', [])
        if not isinstance(iph, list):
            iph = [iph] if iph is not None else []
        iph = iph + ['PYTHON', 'REGISTERED']
        tmpMeta['ImageProcessingHistory'] = iph
        tmpMeta['Keep_image_geometry']    = 1
        if not isinstance(maxVal, np.ndarray):
            tmpMeta['WindowCenter'] = str((int(maxVal) + 1) / 2)
            tmpMeta['WindowWidth']  = str(int(maxVal) + 0)
        if ref_b is not None:
            tmpMeta['RegisteredToBValue'] = str(ref_b)

        imagesOut[iImg].attribute_string = tmpMeta.serialize()

    return imagesOut
