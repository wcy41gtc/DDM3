"""DAS fiber and channel classes for DDM3D calculations."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class Channel:
    """
    A single channel in a DAS fiber optic cable.
    
    This class represents a measurement point along a DAS fiber and stores
    the calculated stress and displacement data at that location.
    """
    
    def __init__(
        self,
        channel_id: int,
        position: Tuple[float, float, float],
        gauge_length: float
    ):
        """
        Initialize a DAS channel.
        
        Parameters
        ----------
        channel_id : int
            Unique identifier for the channel
        position : Tuple[float, float, float]
            Channel position (x, y, z) in meters
        gauge_length : float
            Gauge length in meters
        """
        self._channel_id = int(channel_id)
        
        if len(position) != 3:
            raise ValueError("Position must be a 3-tuple (x, y, z)")
        self._position = tuple(float(p) for p in position)
        
        if gauge_length <= 0:
            raise ValueError("Gauge length must be positive")
        self._gauge_length = float(gauge_length)
        
        # Initialize data storage
        self._stress_data: Dict[str, List[float]] = {
            "SXX": [],
            "SYY": [],
            "SZZ": [],
            "SXY": [],
            "SXZ": [],
            "SYZ": []
        }
        
        self._strain_data: Dict[str, List[float]] = {
            "EXX": [],
            "EYY": [],
            "EZZ": []
        }
        
        self._displacement_data: Dict[str, List[float]] = {
            "UXX": [],
            "UYY": [],
            "UZZ": []
        }
    
    @property
    def channel_id(self) -> int:
        """Channel identifier."""
        return self._channel_id
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Channel position (x, y, z) in meters."""
        return self._position
    
    @property
    def x(self) -> float:
        """X coordinate in meters."""
        return self._position[0]
    
    @property
    def y(self) -> float:
        """Y coordinate in meters."""
        return self._position[1]
    
    @property
    def z(self) -> float:
        """Z coordinate in meters."""
        return self._position[2]
    
    @property
    def gauge_length(self) -> float:
        """Gauge length in meters."""
        return self._gauge_length
    
    def add_stress_data(
        self,
        sxx: float,
        syy: float,
        szz: float,
        sxy: float,
        sxz: float,
        syz: float
    ) -> None:
        """
        Add stress data for this channel.
        
        Parameters
        ----------
        sxx, syy, szz : float
            Normal stress components in Pa
        sxy, sxz, syz : float
            Shear stress components in Pa
        """
        self._stress_data["SXX"].append(float(sxx))
        self._stress_data["SYY"].append(float(syy))
        self._stress_data["SZZ"].append(float(szz))
        self._stress_data["SXY"].append(float(sxy))
        self._stress_data["SXZ"].append(float(sxz))
        self._stress_data["SYZ"].append(float(syz))
    
    def add_strain_data(self, exx: float, eyy: float, ezz: float) -> None:
        """
        Add strain data for this channel.
        
        Parameters
        ----------
        exx, eyy, ezz : float
            Strain components (dimensionless)
        """
        self._strain_data["EXX"].append(float(exx))
        self._strain_data["EYY"].append(float(eyy))
        self._strain_data["EZZ"].append(float(ezz))
    
    def add_displacement_data(self, uxx: float, uyy: float, uzz: float) -> None:
        """
        Add displacement data for this channel.
        
        Parameters
        ----------
        uxx, uyy, uzz : float
            Displacement components in meters
        """
        self._displacement_data["UXX"].append(float(uxx))
        self._displacement_data["UYY"].append(float(uyy))
        self._displacement_data["UZZ"].append(float(uzz))
    
    def get_stress_data(self, component: str) -> List[float]:
        """
        Get stress data for a specific component.
        
        Parameters
        ----------
        component : str
            Stress component ('SXX', 'SYY', 'SZZ', 'SXY', 'SXZ', 'SYZ')
            
        Returns
        -------
        List[float]
            List of stress values
        """
        if component not in self._stress_data:
            raise ValueError(f"Invalid stress component: {component}")
        return self._stress_data[component].copy()
    
    def get_strain_data(self, component: str) -> List[float]:
        """
        Get strain data for a specific component.
        
        Parameters
        ----------
        component : str
            Strain component ('EXX', 'EYY', 'EZZ')
            
        Returns
        -------
        List[float]
            List of strain values
        """
        if component not in self._strain_data:
            raise ValueError(f"Invalid strain component: {component}")
        return self._strain_data[component].copy()
    
    def get_displacement_data(self, component: str) -> List[float]:
        """
        Get displacement data for a specific component.
        
        Parameters
        ----------
        component : str
            Displacement component ('UXX', 'UYY', 'UZZ')
            
        Returns
        -------
        List[float]
            List of displacement values
        """
        if component not in self._displacement_data:
            raise ValueError(f"Invalid displacement component: {component}")
        return self._displacement_data[component].copy()
    
    def clear_data(self) -> None:
        """Clear all stored data."""
        for key in self._stress_data:
            self._stress_data[key].clear()
        for key in self._strain_data:
            self._strain_data[key].clear()
        for key in self._displacement_data:
            self._displacement_data[key].clear()
    
    def get_data_count(self) -> int:
        """Get the number of data points stored."""
        if not self._stress_data["SXX"]:
            return 0
        return len(self._stress_data["SXX"])
    
    def __repr__(self) -> str:
        return (
            f"Channel(id={self._channel_id}, "
            f"position={self._position}, "
            f"gauge_length={self._gauge_length})"
        )
    
    def __str__(self) -> str:
        data_count = self.get_data_count()
        return (
            f"Channel {self._channel_id}:\n"
            f"  Position: {self._position} m\n"
            f"  Gauge Length: {self._gauge_length} m\n"
            f"  Data Points: {data_count}"
        )


class Fiber:
    """
    A DAS fiber optic cable containing multiple measurement channels.
    
    This class represents a complete DAS fiber with its channels and provides
    methods for creating different fiber geometries and accessing channel data.
    """
    
    def __init__(self, fiber_id: int, channels: List[Channel]):
        """
        Initialize a DAS fiber.
        
        Parameters
        ----------
        fiber_id : int
            Unique identifier for the fiber
        channels : List[Channel]
            List of channels in the fiber
        """
        self._fiber_id = int(fiber_id)
        self._channels = list(channels)
        
        if not self._channels:
            raise ValueError("Fiber must contain at least one channel")
    
    @classmethod
    def create_linear(
        cls,
        fiber_id: int,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        n_channels: int
    ) -> "Fiber":
        """
        Create a linear fiber between two points.
        
        Parameters
        ----------
        fiber_id : int
            Unique identifier for the fiber
        start : Tuple[float, float, float]
            Start coordinates (x, y, z) in meters
        end : Tuple[float, float, float]
            End coordinates (x, y, z) in meters
        n_channels : int
            Number of channels in the fiber
            
        Returns
        -------
        Fiber
            Linear fiber instance
        """
        if n_channels <= 0:
            raise ValueError("Number of channels must be positive")
        
        # Calculate channel positions
        start_array = np.array(start)
        end_array = np.array(end)
        total_length = np.linalg.norm(end_array - start_array)
        gauge_length = total_length / n_channels
        
        channels = []
        for i in range(n_channels):
            # Position channels at the center of each segment
            t = (2 * i + 1) / (2 * n_channels)
            position = start_array + t * (end_array - start_array)
            
            channel = Channel(
                channel_id=i + 1,
                position=tuple(position),
                gauge_length=gauge_length
            )
            channels.append(channel)
        
        return cls(fiber_id, channels)
    
    @classmethod
    def create_curved(
        cls,
        fiber_id: int,
        points: List[Tuple[float, float, float]],
        n_channels: int
    ) -> "Fiber":
        """
        Create a curved fiber following a series of points.
        
        Parameters
        ----------
        fiber_id : int
            Unique identifier for the fiber
        points : List[Tuple[float, float, float]]
            List of points defining the fiber path
        n_channels : int
            Number of channels in the fiber
            
        Returns
        -------
        Fiber
            Curved fiber instance
        """
        if len(points) < 2:
            raise ValueError("At least 2 points required for curved fiber")
        if n_channels <= 0:
            raise ValueError("Number of channels must be positive")
        
        # Calculate total path length
        total_length = 0.0
        segment_lengths = []
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            length = np.linalg.norm(end - start)
            segment_lengths.append(length)
            total_length += length
        
        gauge_length = total_length / n_channels
        
        # Distribute channels along the path
        channels = []
        current_length = 0.0
        target_length = gauge_length / 2.0  # First channel at half gauge length
        
        segment_idx = 0
        segment_start = np.array(points[0])
        segment_end = np.array(points[1])
        segment_remaining = segment_lengths[0]
        
        for i in range(n_channels):
            # Find which segment contains the target position
            while target_length > current_length + segment_remaining:
                current_length += segment_remaining
                segment_idx += 1
                if segment_idx >= len(points) - 1:
                    break
                segment_start = np.array(points[segment_idx])
                segment_end = np.array(points[segment_idx + 1])
                segment_remaining = segment_lengths[segment_idx]
            
            # Calculate position within current segment
            if segment_idx < len(points) - 1:
                local_length = target_length - current_length
                t = local_length / segment_remaining
                position = segment_start + t * (segment_end - segment_start)
            else:
                position = np.array(points[-1])
            
            channel = Channel(
                channel_id=i + 1,
                position=tuple(position),
                gauge_length=gauge_length
            )
            channels.append(channel)
            
            target_length += gauge_length
        
        return cls(fiber_id, channels)
    
    @property
    def fiber_id(self) -> int:
        """Fiber identifier."""
        return self._fiber_id
    
    @property
    def channels(self) -> List[Channel]:
        """List of channels in the fiber."""
        return self._channels.copy()
    
    @property
    def n_channels(self) -> int:
        """Number of channels in the fiber."""
        return len(self._channels)
    
    def get_channel(self, channel_id: int) -> Channel:
        """
        Get a specific channel by ID.
        
        Parameters
        ----------
        channel_id : int
            Channel identifier
            
        Returns
        -------
        Channel
            The requested channel
            
        Raises
        ------
        ValueError
            If channel_id is not found
        """
        for channel in self._channels:
            if channel.channel_id == channel_id:
                return channel
        raise ValueError(f"Channel {channel_id} not found")
    
    def get_channel_positions(self) -> List[Tuple[float, float, float]]:
        """
        Get positions of all channels.
        
        Returns
        -------
        List[Tuple[float, float, float]]
            List of channel positions
        """
        return [channel.position for channel in self._channels]
    
    def clear_all_data(self) -> None:
        """Clear data from all channels."""
        for channel in self._channels:
            channel.clear_data()
    
    def get_total_length(self) -> float:
        """
        Calculate total fiber length.
        
        Returns
        -------
        float
            Total fiber length in meters
        """
        if len(self._channels) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self._channels) - 1):
            pos1 = np.array(self._channels[i].position)
            pos2 = np.array(self._channels[i + 1].position)
            total_length += np.linalg.norm(pos2 - pos1)
        
        return total_length
    
    def __repr__(self) -> str:
        return f"Fiber(id={self._fiber_id}, n_channels={self.n_channels})"
    
    def __str__(self) -> str:
        return (
            f"Fiber {self._fiber_id}:\n"
            f"  Channels: {self.n_channels}\n"
            f"  Total Length: {self.get_total_length():.2f} m"
        )
