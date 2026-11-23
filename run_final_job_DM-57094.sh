#!/usr/bin/env bash

python -c "import lsst.resources; print(lsst.resources.__file__, getattr(lsst.resources, '__version__', None))"
eups list -s resources

# step4b
uniqProcName=LSSTCam_runs_DRP_w_2026_43_DM-56803_20261102T001157Z

## step2a_step2c
#uniqProcName=LSSTCam_runs_DRP_w_2026_43_DM-56803_20261028T044141Z

export LSST_SHARED_SCRATCH_DIR=${LSST_RUN_TEMP_SPACE}
export LSST_LOCAL_SCRATCH_DIR=${LSST_GLIDEIN_DIR}

fileDistributionEndpoint="${LSST_SHARED_SCRATCH_DIR}/bps_dev/payload/${uniqProcName}/"

export shared_qg=${fileDistributionEndpoint}${uniqProcName}.qg
echo shared_qg ${shared_qg}
base_qg=`basename ${shared_qg}`;
local_qg=${LSST_LOCAL_SCRATCH_DIR}/${base_qg}
echo local_qg ${local_qg}
if [ -f ${local_qg} ]; then
  echo The file exists.  No copy.
else
  echo The file does not exist.  Using gfal-copy to copy the file
  echo gfal-copy ${shared_qg} ${local_qg}.$$.tmp
  gfal-copy ${shared_qg} ${local_qg}.$$.tmp
  ls -lart ${LSST_LOCAL_SCRATCH_DIR}
  ln ${local_qg}.$$.tmp ${local_qg}
fi
ls -lart ${LSST_LOCAL_SCRATCH_DIR}

qgraphFle=${local_qg}
butlerConfig=dp2_prep

set -xe

export TMPDIR=/dev/shm
export LSST_RESOURCES_TMPDIR=/dev/shm

butler --long-log --log-level=DEBUG \
       aggregate-graph \
       ${local_qg} \
       ${butlerConfig} \
       -j 16 \
       --register-dataset-types \
       --update-output-chain
