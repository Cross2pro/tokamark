"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os.path
import numpy as np
import zarr
import zarr.storage
import fsspec
import s3fs
import pandas as pd
import warnings
from typing import Union, Any
from collections.abc import Mapping
import logging
from pprint import pprint
import time
from posixpath import join as posix_join
from os.path import join as os_join

from MAST_tools.utils.data_utils import ShotInfo, BaseDataSourceType, ZarrStoreType, ZarrFSStoreType, ShotInfoType
from MAST_tools.utils.zip_store import PrefixedZipStore, TarMemberZipStore, load_tar_index
from MAST_tools.utils.general_utils import get_random_string, warning_print
from MAST_tools.utils.path_utils import DEFAULT_SIGNAL_AVAILABILITY_FILE


# ----------------------------------------------------------------------------------------------------------------------

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ----------------------------------------------------------------------------------------------------------------------
# Default values

DEFAULT_BASE_FSSPEC_PROTOCOL = "simplecache"
DEFAULT_TARGET_FSSPEC_PROTOCOL = "s3"
DEFAULT_S3_ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
DEFAULT_S3_MAST_DATASET_PATH = "/mast/tokamark/v1"
DEFAULT_BASE_LOCAL_ZARR_PATH = "/mast/tokamark/v1"  # <- Replace default value if different installation dir is used.

DEFAULT_LOCAL_FLAG_VALUE = False


# ======================================================================================================================
class MASTStorageManager:
    """
    Class with storage management tools for MAST data.

    Attributes
    ----------
    base_fsspec_protocol : str
        Base protocol used by 'fsspec'.
    target_fsspec_protocol : str
        Target filesystem protocol for the selected base 'fsspec' protocol.
    s3_endpoint_url : str
        Endpoint of the cloud S3 bucket used for remote data pulling.
    s3_mast_dataset_path : str
        Path for the target MAST dataset within the configured S3 bucket.
    base_local_zarr_path : str | None
        Local root path used for local data pulling in Zarr format.
    fs_local_fsspec : fsspec.implementations.local.LocalFileSystem
        A LocalFileSystem instance.
    fs_remote_fsspec : Any
        A filesystem instance via fsspec.filesystem.
    fs_remote_s3fs : Any
        A filesystem instance via s3fs.S3FileSystem.
    store_manager_id : str
        User-defined storage manager ID.

    Methods
    -------
    _validate_base_local_zarr_path()
        Check if local path for Zarr database is set, raising SystemError if not.
    _check_local_zarr_database()
        Run access checks for a potential local Zarr database.
    _is_digit(item)
        Check if provided item is of type digit.
    _get_store_from_data_origin(data_origin)
        Auxiliary function to get Zarr store instance from a given data origin.
    _parse_shot_info_dict(shot_info)
        Parse dictionary with shot information.
    _check_shot_id(shot_id)
        Check shot ID.
    _check_list_of_shot_ids(shot_ids)
        Check list of shot IDs.
    _create_fs_remote(library, warn)
        Create filesystem instance either using 'fsspec' or 's3fs' for remote data request.
    _read_fsspec_listdir(path, local)
        Evaluate the list dir method of 'fsspec' filesystem instance on the provided path.
    check_data_origin(data_origin)
        Check MAST data origin.
    list_all_shots(local)
        Get a list of available MAST shot IDs.
    list_shots_by_signal_availability(availability_data_file_path, required_signals)
        List shot IDs following composite condition for signal availability and given availability file.
    get_all_sources(shot_ids, local)
        Return a dictionary with all available sources per shot ID.
    get_all_signals(shot_ids, local, verbose)
        Return a dictionary with all available signals per shot ID.
    make_shot_store(shot_info, verbose)
        Make a Zarr store (either LocalStore or FsspecStore) for a given target shot.
    make_shot_group(data_origin, verbose)
        Make a shot group from data origin (either a Zarr store or shot info).

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        base_fsspec_protocol: str = DEFAULT_BASE_FSSPEC_PROTOCOL,
        target_fsspec_protocol: str = DEFAULT_TARGET_FSSPEC_PROTOCOL,
        s3_endpoint_url: str = DEFAULT_S3_ENDPOINT_URL,
        s3_mast_dataset_path: str = DEFAULT_S3_MAST_DATASET_PATH,
        base_local_zarr_path: str | None = DEFAULT_BASE_LOCAL_ZARR_PATH,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        base_fsspec_protocol: str
            Base protocol used by 'fsspec'. Some supported protocols include:
            `blockcache`:
                - With this option, data is downloaded block-wise.
                - Restrictions:
                    - It has a storage/OS combination which supports sparse files.
                    - The backend implementation uses files which derive from AbstractBufferedFile.
                    - The library you pass the resultant object to accepts generic python file-like objects.
            `filecache`:
                - Works for all file system implementations, and provides a real local file for other libraries to use.
            `simplecache`:
                - Same as `filecache`, except without options for cache expiry and to check original source.
                - Only option guaranteed to be thread/process-safe.
            Full list of supported protocols obtained via `fsspec.available_protocols()`.
            More info available at https://filesystem-spec.readthedocs.io/en/latest/features.html.
            Default: MAST_tools.utils.store_utils.DEFAULT_BASE_FSSPEC_PROTOCOL.
        target_fsspec_protocol : str
            Target filesystem protocol for the selected base 'fsspec' protocol.
            Default: MAST_tools.utils.store_utils.DEFAULT_TARGET_FSSPEC_PROTOCOL.
        s3_endpoint_url : str
            Endpoint of the cloud S3 bucket used for remote data pulling (i.e., for local=False in some methods).
            Default: MAST_tools.utils.store_utils.DEFAULT_S3_ENDPOINT_URL.
        s3_mast_dataset_path : str
            Path for the target MAST dataset within the configured S3 bucket.
            Default: MAST_tools.utils.store_utils.DEFAULT_S3_MAST_DATASET_PATH.
        base_local_zarr_path : str | None
            Local root path used for local data pulling in Zarr format.
            Default: MAST_tools.utils.store_utils.BASE_LOCAL_ZARR_PATH.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Raises
        ------
        FileNotFoundError
            If provided `base_local_zarr_path` directory is not found.

        """

        self.base_fsspec_protocol = base_fsspec_protocol
        self.target_fsspec_protocol = target_fsspec_protocol
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_mast_dataset_path = s3_mast_dataset_path
        self.base_local_zarr_path = base_local_zarr_path

        self._check_local_zarr_database()
        self.fs_local_fsspec = fsspec.filesystem("file")
        self.fs_remote_fsspec = self._create_fs_remote(library="fsspec")
        self.fs_remote_s3fs = self._create_fs_remote(library="s3fs")

        self.store_manager_id = f"store_manager_{get_random_string(4)}"

    # ------------------------------------------------------------------------------------------------------------------
    def _validate_base_local_zarr_path(self):
        """Check if local path for Zarr database is set, raising SystemError if not."""
        if self.base_local_zarr_path is None:
            raise SystemError(
                "No path for local Zarr database was set during the creation of the MASTStorageManager instance."
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _check_local_zarr_database(self):
        """Run access checks for a potential local Zarr database."""

        warning_message = ""
        if self.base_local_zarr_path is None:
            warning_message = (
                "No path for local Zarr database was set during the creation of the MASTStorageManager instance."
            )
        else:
            # Warn about inexistent path
            if not os.path.isdir(self.base_local_zarr_path):
                warning_message = (
                    f"The path `{self.base_local_zarr_path}` for a local Zarr database, which was set during the "
                    f"creation of the MASTStorageManager instance, does not correspond to a valid local directory."
                )
            else:
                zarr_files = [
                    file_ for file_ in os.listdir(self.base_local_zarr_path) if file_.endswith((".zarr", ".zarr.zip"))
                ]
                if len(zarr_files):
                    print(
                        f"[INFO] Local Zarr database identified under `{self.base_local_zarr_path}` with "
                        f"{len(zarr_files)} Zarr files in it."
                    )
                else:
                    # Warn about no Zarr files found
                    warning_message = (
                        f"No Zarr files found under the path `{self.base_local_zarr_path}`, which was set during the "
                        f"creation of the MASTStorageManager instance."
                    )
        if warning_message:
            additional_warning = (
                f"This will cause local pipelines to fail. To avoid this, either provide a valid installation path for "
                f"a local Zarr database, or use default MASTStorageManager settings and install the database under "
                f"the default directory `{DEFAULT_BASE_LOCAL_ZARR_PATH}`."
            )
            warning_print(f"\n{warning_message} {additional_warning}\n")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _is_digit(item: Any) -> bool:
        """
        Check if provided item is of type digit.

        Parameters
        ----------
        item : Any
            An arbitrary input item.

        Returns
        -------
        bool
            True if `item` is digit (either of int type, or of type str representing a digit), False otherwise.

        """

        is_digit = False
        if isinstance(item, str):
            if item.isdigit():
                is_digit = True
        elif isinstance(item, (int, np.int_)):
            is_digit = True

        return is_digit

    # ------------------------------------------------------------------------------------------------------------------
    def _get_store_from_data_origin(
        self,
        data_origin: BaseDataSourceType,
    ) -> ZarrFSStoreType:
        """
        Auxiliary function to get Zarr store instance from a given data origin.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for group creation. It can be a Mapping (dictionary) with shot information (as in the class
            method `self.make_shot_store()`) or a Zarr store (ZarrStoreType instance).

        Returns
        -------
        ZarrFSStoreType
            Instance of Zarr store.

        Raises
        ------
        TypeError
            If `data_origin` is not of type `BaseDataSourceType`.

        """

        self.check_data_origin(data_origin=data_origin)
        if isinstance(data_origin, dict):
            # data_origin is a dict with shot info
            return self.make_shot_store(shot_info=data_origin)  # noqa - Ignore expected type warning
        elif isinstance(data_origin, ZarrFSStoreType):
            # data_origin is a Zarr store
            return data_origin
        else:
            raise TypeError("Invalid type for `data_origin`: it must be of type BaseDataSourceType.")

    # ------------------------------------------------------------------------------------------------------------------
    def _parse_shot_info_dict(
        self,
        shot_info: ShotInfoType,
    ) -> dict[str, Any]:
        """
        Parse dictionary with shot information.

        Parameters
        ----------
        shot_info : ShotInfoType
            Dictionary with shot information required for store creation, with valid keys and types as defined in
            `MAST_tools.data_models.ShotInfo`. Default value for the non-required key "local" is
            `MAST_tools.utils.store_utils.DEFAULT_LOCAL_FLAG_VALUE`.

        Returns
        -------
        dict[str, Any]
            Dictionary with parsed items.

        Raises
        ------
        KeyError
            If field "shot_id" is missing in parameter `shot_info`.
        TypeError
            If field "local" in `shot_info` is not boolean.

        """

        parsed_shot_info = {}

        # Validate "shot_id" field
        if "shot_id" not in shot_info:
            raise KeyError("Missing field `shot_id`.")
        else:
            parsed_shot_info["shot_id"] = shot_info.get("shot_id")
            self._check_shot_id(shot_id=parsed_shot_info["shot_id"])

        # Validate "local" field
        parsed_shot_info["local"] = shot_info.get("local", DEFAULT_LOCAL_FLAG_VALUE)
        if not isinstance(parsed_shot_info["local"], bool):
            raise TypeError("Invalid field `local`: it must be of type bool.")

        return parsed_shot_info

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def check_data_origin(
        data_origin: BaseDataSourceType,
    ) -> None:
        """
        Check MAST data origin.

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Object expected to define data origin for Zarr group/store creation, either Mapping or ZarrStoreType.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If parameter data_origin is not dict or `ZarrStoreType` (i.e., either `zarr.storage.FsspecStore` or
            `zarr.storage.LocalStore`).

        """

        if not isinstance(data_origin, Union[dict, ZarrStoreType]):
            raise TypeError("Invalid parameter `data_origin`: it must be of type dict or ZarrStoreType.")

    # ------------------------------------------------------------------------------------------------------------------
    def _check_shot_id(self, shot_id: Any) -> None:
        """
        Check if provided shot ID has a valid type/value. Allowed types are int, or str with value being a digit.

        Parameters
        ----------
        shot_id : Any
            Target shot ID.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If parameter `shot_id` not of type int or str.
        ValueError
            If parameter `shot_id` is of type str and its value is not a digit.

        """

        if not self._is_digit(item=shot_id):
            raise ValueError(
                "Invalid field/parameter 'shot_id': it must be of type int, or of type str representing a digit."
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _check_list_of_shot_ids(
        self,
        shot_ids: Union[list[int], tuple[int]],
    ) -> None:
        """
        Check list of shot IDs.

        Parameters
        ----------
        shot_ids : Union[list[int], tuple[int]]
            List of shot IDs.

        Returns
        -------
        None

        """

        for id_ in shot_ids:
            self._check_shot_id(shot_id=id_)

    # ------------------------------------------------------------------------------------------------------------------
    def _create_fs_remote(self, library: str, warn: bool = False) -> Any:
        """
        Create a filesystem instance either using 'fsspec' or 's3fs' libraries for remote data request.

        Parameters
        ----------
        library : str
            The target library for filesystem instance creation.
        warn : bool
            If True, warn about issues during filesystem instance creation.
            Default: False.

        Returns
        -------
        Any
            A filesystem instance, created via either `fsspec.filesystem` or `s3fs.S3FileSystem`.

        Raises
        ------
        NotImplementedError
            If a library other than 'fsspec' or 's3fs' is provided.

        """

        if library == "fsspec":
            if warn:
                warnings.warn("WARNING: Library `fsspec` cannot create asynchronous instances of file systems.")

            return fsspec.filesystem(
                protocol=self.base_fsspec_protocol,
                target_protocol=self.target_fsspec_protocol,
                target_options={"anon": True, "endpoint_url": self.s3_endpoint_url},
                # cache_storage='.cache',  # Uncomment and define cache folder, if required.
                # asynchronous=True  # REMARK: This is allowed, but it does not make the instance asynchronous.
            )

        elif library == "s3fs":
            return s3fs.S3FileSystem(anon=True, endpoint_url=self.s3_endpoint_url, asynchronous=True)
        else:
            raise NotImplementedError(f"Error: Library {library} not supported.")

    # ------------------------------------------------------------------------------------------------------------------
    def _read_fsspec_listdir(self, path: str, local: bool = False) -> list:
        """
        Evaluate the list dir method of 'fsspec' filesystem instance on the provided path.

        Parameters
        ----------
        path : str
            Target path to be evaluated via the invoked list dir method.
        local : bool
            Boolean flag to define the 'fsspec' instance to be used. If True, it corresponds to `self.fs_local_fsspec`,
            i.e., a `fsspec.implementations.local.LocalFileSystem` instance; otherwise, `self.fs_remote_fsspec` is used,
            which corresponds to a 'fsspec' instance created either via `fsspec.filesystem` or via `s3fs.S3FileSystem`.
            Default: False.

        Returns
        -------
        list
            List of items in filesystem instance.

        Raises
        ------
        FileNotFoundError
            If invalid path is provided.

        """
        try:
            if local:
                return self.fs_local_fsspec.ls(path)
            else:
                return self.fs_remote_fsspec.ls(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No data available for path {path}.")

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_shots(
        self,
        local: bool = False,
    ) -> list[int]:
        """
        Get a list of available MAST shot IDs.

        Parameters
        ----------
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.

        Returns
        -------
        list[int]
            List of all available shots IDs in the dataset.

        """

        # FSSpec pipeline
        if local:
            self._validate_base_local_zarr_path()
            all_filenames = self._read_fsspec_listdir(path=str(self.base_local_zarr_path), local=True)
        else:
            all_filenames = [
                item["Key"] for item in self._read_fsspec_listdir(path=self.s3_mast_dataset_path, local=False)
            ]

        raw_shot_ids = [
            filename.split("/")[-1].split(".zarr")[0]
            for filename in all_filenames
            if filename.endswith((".zarr", ".zarr.zip"))
        ]
        raw_shot_ids = list(dict.fromkeys(raw_shot_ids))  # dedupe if both .zarr and .zarr.zip exist

        shot_ids = [int(raw_shot_id) for raw_shot_id in raw_shot_ids if self._is_digit(raw_shot_id)]
        shot_ids.sort()

        return shot_ids

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def list_shots_by_signal_availability(
        required_signals: Mapping[str, list[str]], availability_data_file_path: str = DEFAULT_SIGNAL_AVAILABILITY_FILE
    ) -> list:
        """
        List shot IDs following composite condition for signal availability and given availability file.

        Parameters
        ----------
        required_signals: Mapping[str, list[str]]
            Dictionary with required signal availability.
            Example: {"thomson_scattering": ["n_e"], "summary": ["power_nbi", "ip"]}
        availability_data_file_path : str
            Path to suitable CSV file with signal availability.
            Optional. Default: MAST_tools.utils.path_utils.DEFAULT_SIGNAL_AVAILABILITY_FILE.

        Returns
        -------
        list
            List of shot IDs.

        Raises
        ------
        ValueError
            If empty `required_signals` is provided.
        FileNotFoundError
            If invalid `availability_data_file_path` is provided.

        """

        if len(required_signals) == 0:
            raise ValueError("Empty `required_signals` was provided.")

        try:
            availability_data = pd.read_csv(availability_data_file_path)  # noqa - Ignore missing parameter(s) warning
        except FileNotFoundError:
            raise FileNotFoundError(f"Invalid `availability_data_file_path` '{availability_data_file_path}'.")

        source_signals_to_have = []
        for kk, vv in required_signals.items():
            source_signals_to_have += [f"{kk}-{val}" for val in vv]

        composite_condition = availability_data[source_signals_to_have[0]] == True  # noqa - "is" comparison misbehaves
        for signal in source_signals_to_have[1:]:
            composite_condition &= availability_data[signal] == True  # noqa - "is" comparison misbehaves

        return list(availability_data.loc[composite_condition]["shot_id"])  # noqa - Ignore missing attribute warning

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_sources(self, shot_ids: list[int] | None = None, local: bool = False) -> dict[int, list]:
        """
        Return a dictionary with all available sources per shot ID.

        Parameters
        ----------
        shot_ids : list[int] | None
            Target shot IDs to be checked in the MAST database. If None is provided, all available shots are checked.
            Optional. Default: None.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.

        Returns
        -------
        dict[int, list]
            Dictionary with available sources per shot ID.

        """

        if shot_ids is not None:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        # FSSpec pipeline

        if shot_ids is None:
            shot_ids = self.list_all_shots(local=local)

        source_info = {}
        for id_ in shot_ids:
            group = self.make_shot_group(data_origin=ShotInfo(shot_id=id_, local=local))
            source_info[id_] = list(group.keys())

        return source_info

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_signals(
        self, shot_ids: list[int] | None = None, local: bool = False, verbose: bool = False
    ) -> dict[int, list[str]]:
        """
        Return a dictionary with all available signals per shot ID.

        Parameters
        ----------
        shot_ids : list[int] | None
            Target shot IDs to be checked in the MAST database. If None is provided, all shots are checked.
            Optional. Default: None.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.
        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        dict[int, list]
            Dictionary with available signals per shot ID.

        """

        if shot_ids is None:
            shot_ids = self.list_all_shots(local=local)
        else:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        # FSSpec pipeline
        signal_info = {}
        for shot_id in shot_ids:
            group = self.make_shot_group(data_origin=ShotInfo(shot_id=shot_id, local=local))

            group_members = group.members(1)
            for item in group_members:
                member = None
                try:
                    group, member = item[0].split("/")
                    signal_info[shot_id].append(f"{group}-{member}")
                except KeyError:
                    signal_info[shot_id] = [f"{group}-{member}"]
                except ValueError:
                    if verbose:
                        print(f"Skipped item: {item[0]}")

        return signal_info

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_store(self, shot_info: ShotInfoType, verbose: bool = False) -> ZarrStoreType:
        """
        Make a Zarr store (either LocalStore or FsspecStore) for a given target shot.

        Parameters
        ----------
        shot_info : ShotInfoType
            Dictionary with shot information required for store creation, with valid keys and types as defined in
            `MAST_tools.data_models.ShotInfo`. Keys and values are validated via `self._parse_shot_info_dict()`,
            where default values for non-required keys are also set.

        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        ZarrStoreType
            Either a zarr.storage.LocalStore instance or a zarr.storage.FsspecStore instance.

        """

        parsed_shot_info = self._parse_shot_info_dict(shot_info=shot_info)
        if parsed_shot_info["local"]:
            self._validate_base_local_zarr_path()
            store = self._make_local_store(parsed_shot_info["shot_id"])

        else:
            zarr_file_path = posix_join(self.s3_mast_dataset_path, f"{parsed_shot_info['shot_id']}.zarr")
            store = zarr.storage.FsspecStore(fs=self.fs_remote_s3fs, read_only=True, path=zarr_file_path)

        if verbose:
            store_type = "LocalStore" if parsed_shot_info["local"] else "FsspecStore"
            print(f"{store_type} store for shot {parsed_shot_info['shot_id']} created.")

        return store

    # ------------------------------------------------------------------------------------------------------------------
    def _make_local_store(self, shot_id):
        """Local store for a shot. Lookup order: `<shot>.zarr` dir, `<shot>.zarr.zip`, then a zip inside a tar batch
        via `<base>/mast_tar_index.json` (built by tokamark_workspace/build_mast_tar_index.py)."""
        dir_path = os_join(str(self.base_local_zarr_path), f"{shot_id}.zarr")
        if os.path.isdir(dir_path):
            return zarr.storage.LocalStore(root=dir_path, read_only=True)
        zip_path = dir_path + ".zip"
        if os.path.isfile(zip_path):
            return PrefixedZipStore(zip_path, prefix=f"{shot_id}.zarr")
        entry = load_tar_index(self.base_local_zarr_path).get(str(shot_id))
        if entry is not None:
            return TarMemberZipStore(entry[0], entry[1], entry[2], prefix=f"{shot_id}.zarr")
        raise FileNotFoundError(f"Shot {shot_id}: no {dir_path}, {zip_path}, or mast_tar_index.json entry.")

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_group(self, data_origin: BaseDataSourceType, verbose: bool = False) -> zarr.Group:
        """
        Make a shot group from data origin (either a Zarr store or shot info).

        Parameters
        ----------
        data_origin : BaseDataSourceType
            Origin of data for group creation. It can be a Mapping (dictionary) with shot information (as in the class
            method `self.make_shot_store()`) or a Zarr store (ZarrStoreType instance).
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        zarr.Group
            Zarr group created from data origin.

        """

        self.check_data_origin(data_origin=data_origin)
        if isinstance(data_origin, ZarrStoreType):
            # Create group from store
            group = zarr.open_group(store=data_origin, mode="r")  # noqa - Ignore expected type warning
            if verbose:
                print(f"Group for store with path {group.path} created.")
        else:
            # Create group from shot info dictionary

            parsed_shot_info = self._parse_shot_info_dict(shot_info=data_origin)  # noqa - Ignore expected type warning
            if parsed_shot_info["local"]:
                self._validate_base_local_zarr_path()

                # Create group by implicitly creating a writable LocalStore
                group = zarr.open_group(store=self._make_local_store(parsed_shot_info["shot_id"]), mode="r")
                # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation
            else:
                # Create group by implicitly creating a read-only FsspecStore

                remote_shot_path = posix_join(self.s3_mast_dataset_path, f"{parsed_shot_info['shot_id']}.zarr")

                store_path = f"{self.target_fsspec_protocol}:/{remote_shot_path}"
                print(f"store_path: {store_path}")
                group = zarr.open_group(
                    store=store_path, mode="r", storage_options={"anon": True, "endpoint_url": self.s3_endpoint_url}
                )
                # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation

            if verbose:
                print(f"Group for shot {parsed_shot_info['shot_id']} created.")

        if verbose:
            print("Group tree:\n")
            print(group.tree())

        return group

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_all_signals_in_group(group: zarr.Group) -> list[str]:
        """
        Get list of all signals in a given group.

        Parameters
        ----------
        group : zarr.Group
            Zarr group to be inspected for signals.

        Returns
        -------
        list[str]
            List of signals in group using the format [<source>-<signal>].

        """

        full_metadata_dict = group.metadata.to_dict()
        all_metadata_keys = list(full_metadata_dict["consolidated_metadata"]["metadata"].keys())
        found_signals = [kk.replace("/", "-") for kk in all_metadata_keys if "/" in kk]

        return found_signals

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_signals_in_store(self, store: ZarrStoreType) -> list[str]:
        """
        Get list of all signals in a given group.

        Parameters
        ----------
        store : ZarrStoreType
            Zarr store to be inspected for signals.

        Returns
        -------
        list[str]
            List of signals in store using the format [<source>-<signal>].

        """

        found_signals = self.get_all_signals_in_group(group=self.make_shot_group(data_origin=store))

        return found_signals

    # ------------------------------------------------------------------------------------------------------------------
    def are_signals_in_group(self, group: zarr.Group, signals: list[str]) -> dict[str, bool]:
        """
        Evaluate if a given signal is in a target group.

        group :  zarr.Group
            Target group.
        signals: list[str]
            List of signals to be searched within the target group. It expects the format [<source>-<signal>].

        Returns
        -------
        dict[str, bool]
            Dictionary with check results per signal.

        """

        all_signals_in_group = self.get_all_signals_in_group(group=group)
        check_results = {signal_: (signal_ in all_signals_in_group) for signal_ in signals}

        return check_results

    # ------------------------------------------------------------------------------------------------------------------
    def are_signals_in_store(self, store: ZarrStoreType, signals: list[str]) -> dict[str, bool]:
        """
        Evaluate if a given signal is in a target store.

        store : ZarrStoreType
            Target store.
        signals: list[str]
            List of signals to be searched within the target store. It expects the format [<source>-<signal>].

        Returns
        -------
        dict[str, bool]
            Dictionary with check results per signal.

        """

        return self.are_signals_in_group(group=self.make_shot_group(data_origin=store), signals=signals)

    # ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def tests() -> None:
    """
    Quick tests for module functionality.

    Returns
    -------
    None

    """

    t0_all_tests = time.time()

    # ..................................................................................................................

    # Base settings
    shot_ids = [30471]  # None for entire dataset (which takes very long time if local=False).
    shot_id = 30471
    local = True
    verbose = False

    shot_info = ShotInfo(shot_id=shot_id, local=local)

    # Creation of MASTStorageManager instance for test
    store_manager = MASTStorageManager(
        base_fsspec_protocol="simplecache",
        target_fsspec_protocol="s3",
        s3_endpoint_url="https://s3.echo.stfc.ac.uk",
        s3_mast_dataset_path="/mast/tokamark/v1",
        base_local_zarr_path="/mast/tokamark/v1",
    )

    TESTS_TO_RUN = {  # noqa - Ignore lowercase warning
        "get_all_shot_ids": False,
        "get_all_sources": False,
        "get_all_signals": False,
        "make_group_from_store": False,
        "make_group_from_shot_info": False,
        "check_signal_in_store": True,
        "get_all_signals_in_store": True,
        "check_signal_availability": False,
    }

    # ..................................................................................................................
    # List all shot IDs for the entire dataset, using different pipelines

    if TESTS_TO_RUN["get_all_shot_ids"]:
        pipeline_tag = "ffspec"
        local_tag = "local" if local else "remote"  # NOSONAR - Ignore weak warning
        print(f"Getting available shot IDs ({pipeline_tag} pipeline, {local_tag} bucket)...\n")
        t0 = time.time()
        all_shots_ids = store_manager.list_all_shots(local=local)

        print(f"Number of shots: {len(list(all_shots_ids))}")
        print(f"Shot IDs: {list(all_shots_ids)}")
        print(f"Elapsed time: {round(time.time() - t0, 2)} s\n")

    # ..................................................................................................................
    # List all sources

    if TESTS_TO_RUN["get_all_sources"]:
        all_sources = store_manager.get_all_sources(
            shot_ids=shot_ids,
            local=local,
        )
        print(f"all_sources[shot_ids[0]]: {all_sources[shot_ids[0]]}\n")

    # ..................................................................................................................
    # List all signals

    if TESTS_TO_RUN["get_all_signals"]:
        all_signals = store_manager.get_all_signals(shot_ids=shot_ids, local=local, verbose=verbose)
        print(f"all_signals[shot_ids[0]]: {all_signals[shot_ids[0]]}\n")

    # ..................................................................................................................
    # Make group for a given shot via existing store object

    if TESTS_TO_RUN["make_group_from_store"]:
        store_ = store_manager.make_shot_store(shot_info=shot_info)
        group_from_store = store_manager.make_shot_group(data_origin=store_)

        # Print group metadata:
        # print(f"group_from_store.metadata.to_dict() (group from store): {group_from_store.metadata.to_dict()}")

        # Print group tree:
        print(f"group_from_store.tree() (group from store): {group_from_store.tree()}\n")

    # ..................................................................................................................
    # Make group for a given shot directly from shot_info

    if TESTS_TO_RUN["make_group_from_shot_info"]:
        group_from_shot_id = store_manager.make_shot_group(data_origin=shot_info)

        # Print group tree:
        print(f"group_from_shot_id.tree() (group from shot ID): {group_from_shot_id.tree()}\n")

    # ..................................................................................................................
    # Get all the signals available for a given store

    if TESTS_TO_RUN["get_all_signals_in_store"]:
        store_ = store_manager.make_shot_store(shot_info=shot_info)

        all_signals = store_manager.get_all_signals_in_store(store=store_)
        print("All signals in store:")
        pprint(all_signals)

    # ..................................................................................................................
    # Check if signal is in store

    if TESTS_TO_RUN["check_signal_in_store"]:
        store_ = store_manager.make_shot_store(shot_info=shot_info)

        signals_ = ["abc", "summary-power_radiated"]

        check_results = store_manager.are_signals_in_store(store=store_, signals=signals_)
        print("\nCheck results for signals in store:")
        pprint(check_results)

    # ..................................................................................................................
    # List all shots IDs for given signal availability

    if TESTS_TO_RUN["check_signal_availability"]:
        dict_target_signals = {
            "thomson_scattering": ["n_e"],
            "spectrometer_visible": ["filter_spectrometer_bes_voltage"],
            "summary": ["power_nbi", "ip"],
        }
        signal_availability_file = DEFAULT_SIGNAL_AVAILABILITY_FILE

        print("\nSignals to be checked for simultaneous availability across all shots:")
        pprint(dict_target_signals)

        filtered_ids = store_manager.list_shots_by_signal_availability(
            required_signals=dict_target_signals, availability_data_file_path=signal_availability_file
        )

        print(f"\nfiltered_ids ({len(list(filtered_ids))} shots):")
        pprint(filtered_ids)

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
