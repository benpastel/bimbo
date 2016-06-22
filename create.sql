PRAGMA foreign_keys = ON;

drop table products;
create table products(
	Producto_ID INTEGER primary key,
	NombreProducto TEXT not null
);

drop table clients;
create table clients(
	Cliente_ID INTEGER primary key,
	NombreCliente TEXT not null
);

drop table train;
create table train(
	Semana INTEGER not null,
	Agencia_ID INTEGER not null check (Agencia_ID > 0),
	Canal_ID INTEGER not null check (Canal_ID > 0),
	Ruta_SAK INTEGER not null check (Ruta_SAK > 0),
	Cliente_ID references clients,
	Producto_ID references products,
	Venta_uni_hoy INTEGER not null check (Venta_uni_hoy >= 0),
	Venta_hoy REAL not null check (Venta_hoy >= 0),
	Dev_uni_proxima INTEGER not null check (Dev_uni_proxima >= 0),
	Dev_proxima REAL not null check (Dev_proxima >= 0),
	Demanda_uni_equil INTEGER not null check (Demanda_uni_equil >= 0)
);

.mode 'csv'
.import 'data/products.csv' products
.import 'data/clients.csv' clients
.import 'data/train.csv' train
