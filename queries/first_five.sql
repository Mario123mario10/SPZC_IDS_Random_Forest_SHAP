SELECT
    TOP 5 *
FROM
    [1]
WHERE
    label = "DDoS"
UNION ALL
SELECT
    TOP 5 *
FROM
    [3]
WHERE
    label = "BENIGN";