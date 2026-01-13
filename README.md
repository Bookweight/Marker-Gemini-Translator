LIST
FROM &quot;Papers&quot;
LIMIT 5

TABLE field as &quot;領域&quot;, created_date as &quot;日期&quot;, pdf_link as &quot;PDF&quot;
FROM &quot;Papers&quot;
WHERE tags AND contains(tags, &quot;paper&quot;)
SORT created_date DESC