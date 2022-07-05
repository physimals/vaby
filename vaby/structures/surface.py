"""
VABY - Surface structure models
"""
from types import SimpleNamespace

import numpy as np
import nibabel as nib
import tensorflow as tf

try:
    import toblerone
    import regtricks
    from toblerone.classes import Hemisphere, Surface
except ImportError:
    toblerone = None

from ..utils import NP_DTYPE
from .base import DataStructure

class SimpleSurface(DataStructure):
    """
    A surface data structure

    The surface consists of a triangular mesh, the vertices of which are the structure nodes.

    Currently a simple surface cannot be projected onto a volume, however it is possible
    to model surface based acquisition data on a matching surface.
    """
    def __init__(self, data, trigs=None, verts=None, geomdata=None, file_ext=".func.gii", **kwargs):
        DataStructure.__init__(self, file_ext=file_ext, **kwargs)
        if toblerone is None:
            raise RuntimeError("Toblerone not installed - cannot create cortical surface structure")

        self.log.info("Simple surface structure")
        self.num_strucs = 1
        self.srcdata = SimpleNamespace()
        if geomdata is not None and (trigs is not None or verts is not None):
            raise ValueError("Can't specify trigs/vertices and GII file at the same time")
        elif geomdata is not None:
            if isinstance(geomdata, str):
                self.srcdata.geom = Surface(geomdata, "surface")
            else:
                raise NotImplementedError("Can't yet create Toblerone surface from GII structure")
        elif trigs is None or verts is None:
            raise ValueError("Must specify both vertices and triangle lists")
        else:
            self.srcdata.geom = Surface.manual(verts, trigs, name="surface")

        self.srcdata.flat = data.astype(NP_DTYPE)
        while self.srcdata.flat.ndim < 2:
            # Interpret 1D data as 1 node timeseries
            self.srcdata.flat = self.srcdata.flat[np.newaxis, ...]
        if self.srcdata.flat.ndim != 2:
            raise ValueError("Data must be 1d (single node timeseries) or 2D (Timeseries for each node)")
        self.size = self.srcdata.flat.shape[0]
        self.srcdata.n_tpts = self.srcdata.flat.shape[1]
        if self.size != self.srcdata.geom.n_points:
            raise ValueError(f"Timeseries data defined on {self.size} nodes, but {self.srcdata.geom.n_points} nodes in surface geometry data")
        self.log.info(f" - {self.size} vertices")
        self.log.info(f" - {len(self.srcdata.geom.tris)} triangles")
        self.log.info(f" - Source data contained {self.srcdata.n_tpts} time points")

        self.adj_matrix = self.srcdata.geom.adjacency_matrix()
        #self.laplacian = self._scipy_to_tf_sparse(self.srcdata.geom.mesh_laplacian())
        self.laplacian = self.srcdata.geom.mesh_laplacian()

    def model2data(self, tensor, data_space):
        try:
            self.check_compatible(data_space)
            return tensor
        except:
            raise NotImplementedError("Projection between different simple surface spaces")

    def data2model(self, tensor, data_space):
        try:
            self.check_compatible(data_space)
            return tensor
        except:
            raise NotImplementedError("Projection between different simple surface spaces")

    def load_data(self, data, **kwargs):
        from . import get_data_structure # Avoid circular import
        data_struc = get_data_structure(data, **kwargs)
        data_struc.check_compatible(self)
        return data_struc.srcdata.flat

    def nibabel_image(self, data):
        if data.shape[0] != self.size:
            raise ValueError(f"Data defined on {data.shape[0]} nodes, this structure has {self.size} nodes")
        
        meta = {'Description': f'{self.name} surface parameter estimates produced by vaby'}
        arr = nib.gifti.GiftiDataArray(
            data.astype(np.float32),
            intent='NIFTI_INTENT_ESTIMATE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=meta)
        return nib.GiftiImage(darrays=[arr])

class CorticalSurface(DataStructure):
    """
    A cortical surface data structure

    This consists two surfaces, inner (white matter) and outer (pial). These surfaces
    contain corresponding tri-meshes that define triangular voxels each of which represents
    a node in the structure.

    Multiple cortical surface structures can be combined in a CompositeStructure, e.g.
    right and left hemispheres, potentially additionally with volumetric subcortical white matter
    """

    def __init__(self, white, pial, file_ext=".func.gii", **kwargs):
        """
        :param left: Tuple of (LWS, LPS) where LWS and LPS are filenames or nibabel.Gifti1Image instances
                     containing the white and pial surfaces for the left hemisphere
        :param right: Tuple of (RWS, RPS) where LWS and LPS are filenames or nibabel.Gifti1Image instances
                      containing the white and pial surfaces for the right hemisphere
        """
        DataStructure.__init__(self, file_ext=file_ext, **kwargs)
        if toblerone is None:
            raise RuntimeError("Toblerone not installed - cannot create cortical surface structure")

        if self.name not in ("L", "R"):
            raise ValueError("For now, the names of cortical surfaces must be either 'L' or 'R'")

        self.log.info("Cortical surface structure")
        self.hemisphere = Hemisphere(Surface(white, self.name + "WS"), Surface(pial, self.name + "PS"), self.name)
        self.size = self.hemisphere.n_points

        self.log.info(f" - {self.size} vertices")
        self.projector = kwargs.get("projector", None)
        if isinstance(self.projector, str):
            self.log.info(f" - Loading projector from {self.projector}")
            self.projector = toblerone.Projector.load(self.projector)

        self.adj_matrix = self.hemisphere.adjacency_matrix()
        self.laplacian = self.hemisphere.mesh_laplacian()

    def _generate_projector(self, data_space):
        if self.projector is None:
            self.log.info("Generating projector - this may take some time...")
            self.projector = toblerone.Projector(self.hemisphere, regtricks.ImageSpace(data_space.srcdata.nii), factor=10, cores=8)
            self.projector.save("vaby_proj.h5")
            self.log.info("Projector generated")

    def model2data(self, tensor, data_space):
        self._generate_projector(data_space)
        n2v = self.projector.surf2vol_matrix(edge_scale=True).astype(NP_DTYPE)

        #if data_space.size != n2v.shape[0]:
        #    raise ValueError('Acquisition data size does not match projector')
        if self.size != n2v.shape[1]:
            raise ValueError('Model size does not match projector')

        # Knock out voxels from projection matrices that are not in the mask
        vox_inds = np.flatnonzero(data_space.mask)
        masked_mat = n2v.tocsr()[vox_inds, :].tocoo()
        proj_tensor = tf.SparseTensor(
            indices=np.array([masked_mat.row, masked_mat.col]).T,
            values=masked_mat.data,
            dense_shape=masked_mat.shape,
        )

        return tf.sparse.sparse_dense_matmul(proj_tensor, tensor)

    def data2model(self, tensor, data_space):
        self._generate_projector(data_space)

        v2n = self.projector.vol2surf_matrix(edge_scale=False).astype(NP_DTYPE)
        #if tf.shape(tensor)[0] != v2n.shape[1]:
        #    raise ValueError('Tensor size does not match projector')
        #if data_space.size != v2n.shape[1]:
        #    raise ValueError('Acquisition data size does not match projector')
        if self.size != v2n.shape[0]:
            raise ValueError('Model size does not match projector')

        # Knock out voxels from projection matrices that are not in the mask
        vox_inds = np.flatnonzero(data_space.mask)
        masked_mat = v2n.tocsr()[:, vox_inds].tocoo()
        proj_tensor = tf.SparseTensor(
            indices=np.array([masked_mat.row, masked_mat.col]).T,
            values=masked_mat.data,
            dense_shape=masked_mat.shape,
        )

        return tf.sparse.sparse_dense_matmul(proj_tensor, tensor)

    def load_data(self, data, **kwargs):
        raise NotImplementedError()

    def nibabel_image(self, data):
        if data.shape[0] != self.size:
            raise ValueError("Incorrect data shape for surface")

        meta = {'Description': f'{self.name} cortex parameter estimates produced by vaby'}
        arr = nib.gifti.GiftiDataArray(
            data.astype(np.float32),
            intent='NIFTI_INTENT_ESTIMATE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=meta)
        return nib.GiftiImage(darrays=[arr])
