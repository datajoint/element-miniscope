import datajoint as dj

from . import miniscope_no_curation as miniscope

schema = dj.schema()


def activate(
    schema_name, *, create_schema=True, create_tables=True
):
    """Activate this schema.

    The "activation" of miniscope_report should be evoked by the miniscope module

    Args:
        schema_name (str): schema name on the database server to activate the
            `miniscope_report` schema
        create_schema (bool): when True (default), create schema in the database if it
            does not yet exist.
        create_tables (str): when True (default), create schema takes in the database
            if they do not yet exist.
    """
    if not miniscope.schema.is_activated():
        raise RuntimeError("Please activate the `miniscope` schema first.")

    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=miniscope.__dict__,
    )


@schema
class QualityMetrics(dj.Imported):
    definition = """
    -> miniscope.Processing
    ---
    r_values=null  : longblob # space correlation for each component
    snr=null       : longblob # trace SNR for each component
    cnn_preds=null : longblob # CNN predictions for each component
    """

    def make(self, key):
        from .miniscope import get_loader_result

        method, loaded_result = get_loader_result(key, miniscope.Curation)
        assert (
            method == "caiman"
        ), f"Quality figures for {method} not yet implemented. Try CaImAn."

        key.update(
            {
                attrib_name: getattr(loaded_result.cnmf.estimates, attrib, None)
                for attrib_name, attrib in zip(
                    ["r_values", "snr", "cnn_preds"],
                    ["r_values", "SNR_comp", "cnn_preds"],
                )
            }
        )

        self.insert1(key)
