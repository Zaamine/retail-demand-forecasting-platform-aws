-- Return duplicate date-store keys; the test passes only when this query returns zero rows.
select
    date,
    store_nbr,
    count(*) as row_count
from {{ ref('stg_transactions') }}
group by
    date,
    store_nbr
having count(*) > 1
