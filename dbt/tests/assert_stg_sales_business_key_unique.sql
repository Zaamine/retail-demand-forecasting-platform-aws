-- Return duplicate business keys; the test passes only when this query returns zero rows.
select
    date,
    store_nbr,
    family,
    count(*) as row_count
from {{ ref('stg_sales') }}
group by
    date,
    store_nbr,
    family
having count(*) > 1
