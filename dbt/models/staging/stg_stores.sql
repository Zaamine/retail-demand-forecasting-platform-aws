select
    cast(store_nbr as bigint) as store_nbr,
    city,
    state,
    type,
    cast(cluster as bigint) as cluster
from {{ source('raw', 'stores') }}
