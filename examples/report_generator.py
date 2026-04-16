"""Report Generation - PDF and CSV export utilities."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any

from event_service import DetectionEvent


class ReportGenerator:
    """Generate reports from event data."""

    @staticmethod
    def generate_csv(events: list[DetectionEvent], camera_id: str | None = None) -> str:
        """Generate CSV report from events."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Timestamp",
            "Camera ID",
            "Track ID",
            "Event Type",
            "Class Name",
            "Confidence",
            "X",
            "Y",
            "X Min",
            "Y Min",
            "X Max",
            "Y Max",
            "Region Name",
        ])

        # Data rows
        for event in events:
            writer.writerow([
                event.timestamp.isoformat(),
                event.camera_id,
                event.track_id,
                event.event_type.value,
                event.class_name,
                f"{event.confidence:.4f}",
                f"{event.x:.2f}",
                f"{event.y:.2f}",
                f"{event.x_min:.2f}",
                f"{event.y_min:.2f}",
                f"{event.x_max:.2f}",
                f"{event.y_max:.2f}",
                event.region_name or "N/A",
            ])

        return output.getvalue()

    @staticmethod
    def generate_csv_summary(stats: dict[str, Any]) -> str:
        """Generate CSV summary report from statistics."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Camera ID",
            "Total Tracks",
            "Entries",
            "Exits",
            "Current Crowd",
            "Unique IDs",
            "Last Updated",
        ])

        # Data rows
        for camera_id, cam_stats in stats.items():
            writer.writerow([
                camera_id,
                cam_stats.get("total_tracks", 0),
                cam_stats.get("entry_count", 0),
                cam_stats.get("exit_count", 0),
                cam_stats.get("current_crowd", 0),
                cam_stats.get("unique_ids_count", 0),
                cam_stats.get("last_updated", "N/A"),
            ])

        return output.getvalue()

    @staticmethod
    def generate_pdf_summary(stats: dict[str, Any], filename: str = "report.pdf") -> None:
        """Generate PDF report from statistics."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.units import inch
            from reportlab.lib import colors

            doc = SimpleDocTemplate(filename, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Title
            title = Paragraph(f"<b>Event Tracking Report</b><br/><font size=10>{datetime.now().isoformat()}</font>", styles["Heading1"])
            story.append(title)
            story.append(Spacer(1, 0.5 * inch))

            # Summary table
            summary_data = [["Camera ID", "Entries", "Exits", "Current Crowd", "Unique IDs"]]
            for camera_id, cam_stats in stats.items():
                summary_data.append([
                    camera_id,
                    str(cam_stats.get("entry_count", 0)),
                    str(cam_stats.get("exit_count", 0)),
                    str(cam_stats.get("current_crowd", 0)),
                    str(cam_stats.get("unique_ids_count", 0)),
                ])

            table = Table(summary_data)
            table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ])
            )
            story.append(table)

            doc.build(story)
            print(f"PDF report generated: {filename}")

        except ImportError:
            print("reportlab not installed. Install with: pip install reportlab")

    @staticmethod
    def export_json(events: list[DetectionEvent]) -> str:
        """Export events as JSON string."""
        import json

        data = [event.to_dict() for event in events]
        return json.dumps(data, indent=2)


# Convenience functions
def export_events_csv(events: list[DetectionEvent], filename: str = "events.csv") -> str:
    """Export events to CSV file."""
    csv_content = ReportGenerator.generate_csv(events)
    with open(filename, "w", newline="") as f:
        f.write(csv_content)
    print(f"CSV exported to: {filename}")
    return filename


def export_summary_csv(stats: dict[str, Any], filename: str = "summary.csv") -> str:
    """Export summary statistics to CSV file."""
    csv_content = ReportGenerator.generate_csv_summary(stats)
    with open(filename, "w", newline="") as f:
        f.write(csv_content)
    print(f"Summary CSV exported to: {filename}")
    return filename


def export_pdf(stats: dict[str, Any], filename: str = "report_{}.pdf") -> str:
    """Export report to PDF file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = filename.format(timestamp)
    ReportGenerator.generate_pdf_summary(stats, pdf_filename)
    return pdf_filename
