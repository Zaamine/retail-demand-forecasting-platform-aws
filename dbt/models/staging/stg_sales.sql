select
    cast(id as bigint) as id,
    cast(date as date) as date,
    cast(store_nbr as bigint) as store_nbr,
    family,
    cast(sales as double) as sales,
    cast(onpromotion as bigint) as onpromotion
from {{ source('raw', 'train') }}
