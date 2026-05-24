"""Tag grid / scene description."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from render_model.tag_data import TAG_CANONICAL_CORNERS, TAG_GRID_SIZE, build_tag_pattern


@dataclass(frozen=True)
class TagInstance:
    """Single tag in the scene."""

    tag_id: int
    # Top-left corner of tag in scene coordinates (tag pixel units).
    origin: tuple[float, float] = (0.0, 0.0)


@dataclass
class TagGrid:
    """Scene containing one or more AprilTag instances."""

    tags: tuple[TagInstance, ...] = (TagInstance(tag_id=0),)
    grid_size: int = TAG_GRID_SIZE
    background: float = 1.0  # white

    def canonical_corners(self, tag: TagInstance | None = None) -> jnp.ndarray:
        """Unit tag corners (0..grid_size) for the first tag or given instance."""
        tag = tag or self.tags[0]
        ox, oy = tag.origin
        c = TAG_CANONICAL_CORNERS.copy()
        c[:, 0] += ox
        c[:, 1] += oy
        return jnp.asarray(c, dtype=jnp.float32)

    def tag_pattern(self, tag_id: int = 0) -> jnp.ndarray:
        return build_tag_pattern(tag_id)

    def composite_pattern(self) -> jnp.ndarray:
        """Flat grid_size x grid_size pattern for the default single-tag scene."""
        if len(self.tags) != 1:
            raise NotImplementedError("Composite pattern only for single-tag PoC")
        return self.tag_pattern(self.tags[0].tag_id)


@dataclass
class RenderScene:
    """Full scene + image dimensions for rendering."""

    tag_grid: TagGrid
    image_width: int
    image_height: int
    # Ground-truth image-space corners (TL, TR, BR, BL).
    gt_corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))

    @property
    def src_corners(self) -> jnp.ndarray:
        return self.tag_grid.canonical_corners()
