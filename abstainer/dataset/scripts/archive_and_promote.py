#!/usr/bin/env python3
"""
Archive current TSV files and report files, then promote curated_questions.tsv to questions.tsv

This script:
1. Creates a 'zero_pass' archive folder in the dataset directory
2. Creates 'data' and 'reports' subfolders within the archive
3. Copies all current TSV files to zero_pass/data/
4. Copies all report files (if they exist) to zero_pass/reports/
5. Copies curated_questions.tsv to questions.tsv (promoting it as the new main file)
6. Removes all original files except the newly promoted questions.tsv
"""

import os
import shutil
from pathlib import Path


def main():
    # Get the dataset, data and reports directory paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    data_dir = dataset_dir / "data"
    reports_dir = dataset_dir / "reports"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return False
    
    # Create archive directory structure
    archive_dir = dataset_dir / "zero_pass"
    archive_data_dir = archive_dir / "data"
    archive_reports_dir = archive_dir / "reports"
    
    archive_dir.mkdir(exist_ok=True)
    archive_data_dir.mkdir(exist_ok=True)
    print(f"Created archive directory: {archive_dir}")
    print(f"Created archive data directory: {archive_data_dir}")
    
    # Find all TSV files in the data directory
    tsv_files = list(data_dir.glob("*.tsv"))
    
    if not tsv_files:
        print("No TSV files found in data directory")
        return False
    
    print(f"\nFound {len(tsv_files)} TSV files to archive:")
    for tsv_file in tsv_files:
        print(f"  - {tsv_file.name}")
    
    # Check for reports directory and files
    report_files = []
    if reports_dir.exists():
        report_files = [f for f in reports_dir.iterdir() if f.is_file()]
        if report_files:
            archive_reports_dir.mkdir(exist_ok=True)
            print(f"Created archive reports directory: {archive_reports_dir}")
            print(f"\nFound {len(report_files)} report files to archive:")
            for report_file in report_files:
                print(f"  - {report_file.name}")
    
    # Archive all TSV files to data subfolder
    print("\nArchiving TSV files...")
    for tsv_file in tsv_files:
        archive_path = archive_data_dir / tsv_file.name
        shutil.copy2(tsv_file, archive_path)
        print(f"  ✓ Archived: {tsv_file.name} -> zero_pass/data/{tsv_file.name}")
    
    # Archive report files to reports subfolder if they exist
    if report_files:
        print("\nArchiving report files...")
        for report_file in report_files:
            archive_path = archive_reports_dir / report_file.name
            shutil.copy2(report_file, archive_path)
            print(f"  ✓ Archived: {report_file.name} -> zero_pass/reports/{report_file.name}")
    
    # Check if curated_questions.tsv exists
    curated_file = data_dir / "curated_questions.tsv"
    if not curated_file.exists():
        print(f"\nError: curated_questions.tsv not found at {curated_file}")
        return False
    
    # Promote curated_questions.tsv to questions.tsv
    new_questions_file = data_dir / "questions.tsv"
    shutil.copy2(curated_file, new_questions_file)
    print(f"\n✓ Promoted: curated_questions.tsv -> questions.tsv")
    
    # Remove original files after archiving (except the promoted questions.tsv)
    print("\nRemoving original files...")
    files_removed = 0
    
    # Remove TSV files (except the newly promoted questions.tsv)
    for tsv_file in tsv_files:
        if tsv_file.name != "questions.tsv":  # Don't remove the newly promoted file
            tsv_file.unlink()
            print(f"  ✓ Removed: {tsv_file.name}")
            files_removed += 1
    
    # Remove report files
    for report_file in report_files:
        report_file.unlink()
        print(f"  ✓ Removed: {report_file.name}")
        files_removed += 1
    
    # Summary
    total_archived = len(tsv_files) + len(report_files)
    print(f"\n{'='*50}")
    print("ARCHIVE, CLEANUP, AND PROMOTION COMPLETE")
    print(f"{'='*50}")
    print(f"Archive location: {archive_dir}")
    print(f"  - Data files: {archive_data_dir}")
    if report_files:
        print(f"  - Report files: {archive_reports_dir}")
    print(f"Archived TSV files: {len(tsv_files)}")
    if report_files:
        print(f"Archived report files: {len(report_files)}")
    print(f"Total archived files: {total_archived}")
    print(f"Files removed: {files_removed}")
    print(f"New main file: questions.tsv (from curated_questions.tsv)")
    print(f"Remaining files in data/: questions.tsv only")
    print(f"{'='*50}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
