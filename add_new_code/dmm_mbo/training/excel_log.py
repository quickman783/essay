"""把每一轮的评估指标追加到同一个 xlsx 表格里。

设计说明：
    xlsx 用标准库直接生成（zipfile + XML），不依赖 openpyxl / pandas，
    免去额外安装。真正的数据源是旁边的同名 .csv，每次追加后由 csv 重新
    生成 xlsx —— 这样即使表格被 Excel 占用打不开，数据也不会丢。

用法：
    append_row(xlsx_path, header, {"轮次": 1, "val_mAP": 0.61, ...})
"""

import csv
import os
import re
import zipfile
from xml.sax.saxutils import escape

# 数字列保留几位小数
_FLOAT_DIGITS = 4

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""

_WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="{sheet}" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""


def _column_name(index):
    """1 -> A, 27 -> AA"""
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(ref, value):
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            value = round(value, _FLOAT_DIGITS)
        text = repr(value) if isinstance(value, float) else str(value)
        return '<c r="{0}"><v>{1}</v></c>'.format(ref, escape(text))
    return '<c r="{0}" t="inlineStr"><is><t xml:space="preserve">{1}</t></is></c>'.format(
        ref, escape(str(value)))


def _sheet_xml(header, rows):
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
             '<sheetData>']
    for row_index, row in enumerate([header] + rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            cell = _cell_xml("{0}{1}".format(_column_name(col_index), row_index), value)
            if cell:
                cells.append(cell)
        lines.append('<row r="{0}">{1}</row>'.format(row_index, "".join(cells)))
    lines.append('</sheetData>')
    lines.append('</worksheet>')
    return "".join(lines)


def _write_xlsx(path, header, rows, sheet_name):
    sheet = _sheet_xml(header, rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _ROOT_RELS)
        archive.writestr("xl/workbook.xml", _WORKBOOK.format(sheet=escape(sheet_name)))
        archive.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def _read_csv(path):
    """返回 (header, rows)，rows 是字符串列表的列表。"""
    if not os.path.isfile(path):
        return [], []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _to_number(text):
    if text == "":
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def append_row(xlsx_path, header, values, csv_path=None, sheet_name="训练记录"):
    """把一行指标追加进表格；表头有新增列时自动补列。

    Args:
        xlsx_path (str): xlsx 文件路径，不存在会新建
        header (list[str]): 本次数据的列名
        values (dict): 列名 -> 值
        csv_path (str, optional): 同源 csv，默认与 xlsx 同名
        sheet_name (str): 工作表名

    Returns:
        str: 实际写入的 xlsx 路径
    """
    csv_path = csv_path or os.path.splitext(xlsx_path)[0] + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(xlsx_path)) or ".", exist_ok=True)

    existing_header, existing_rows = _read_csv(csv_path)
    if not existing_header:
        existing_header = list(header)

    # 追加新列，保持已有列的顺序不变
    for name in header:
        if name not in existing_header:
            existing_header.append(name)

    existing_rows.append([_num_to_text(values.get(name, "")) for name in existing_header])

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(existing_header)
        writer.writerows(existing_rows)

    rows = [[_to_number(text) for text in row] for row in existing_rows]
    _write_xlsx(xlsx_path, existing_header, rows, sheet_name)
    return xlsx_path


def _num_to_text(value):
    if isinstance(value, float):
        return "{0:.{1}f}".format(value, _FLOAT_DIGITS)
    return "" if value is None else str(value)


def sanitize(name):
    """列名里去掉换行等会破坏表格的字符。"""
    return re.sub(r"[\r\n\t]+", " ", str(name)).strip()
